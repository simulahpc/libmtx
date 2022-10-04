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
 * Last modified: 2022-07-11
 *
 * Data structures for vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/blas/vector.h>
#include <libmtx/linalg/omp/vector.h>
#include <libmtx/linalg/local/vector.h>

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
 * Vector types
 */

/**
 * ‘mtxvectortype_str()’ is a string representing the vector type.
 */
const char * mtxvectortype_str(
    enum mtxvectortype type)
{
    switch (type) {
    case mtxbasevector: return "base";
    case mtxblasvector: return "blas";
    case mtxnullvector: return "null";
    case mtxompvector: return "omp";
    default: return mtxstrerror(MTX_ERR_INVALID_VECTOR_TYPE);
    }
}

/**
 * ‘mtxvectortype_parse()’ parses a string to obtain one of the
 * vector types of ‘enum mtxvectortype’.
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
 * On success, ‘mtxvectortype_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘vector_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxvectortype_parse(
    enum mtxvectortype * vector_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("base", t, strlen("base")) == 0) {
        t += strlen("base");
        *vector_type = mtxbasevector;
    } else if (strncmp("blas", t, strlen("blas")) == 0) {
        t += strlen("blas");
        *vector_type = mtxblasvector;
    } else if (strncmp("null", t, strlen("null")) == 0) {
        t += strlen("null");
        *vector_type = mtxnullvector;
    } else if (strncmp("omp", t, strlen("omp")) == 0) {
        t += strlen("omp");
        *vector_type = mtxompvector;
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_VECTOR_TYPE;
        t++;
    }
    if (bytes_read) *bytes_read += t-s;
    if (endptr) *endptr = t;
    return MTX_SUCCESS;
}

/*
 * vector properties
 */

/**
 * ‘mtxvector_field()’ gets the field of a vector.
 */
int mtxvector_field(
    const struct mtxvector * x,
    enum mtxfield * field)
{
    if (x->type == mtxbasevector) {
        *field = x->storage.base.field;
        return MTX_SUCCESS;
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        *field = x->storage.blas.base.field;
        return MTX_SUCCESS;
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        *field = mtxnullvector_field(&x->storage.null);
        return MTX_SUCCESS;
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        *field = x->storage.omp.base.field;
        return MTX_SUCCESS;
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_precision()’ gets the precision of a vector.
 */
int mtxvector_precision(
    const struct mtxvector * x,
    enum mtxprecision * precision)
{
    if (x->type == mtxbasevector) {
        *precision = x->storage.base.precision;
        return MTX_SUCCESS;
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        *precision = x->storage.blas.base.precision;
        return MTX_SUCCESS;
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        *precision = mtxnullvector_precision(&x->storage.null);
        return MTX_SUCCESS;
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        *precision = x->storage.omp.base.precision;
        return MTX_SUCCESS;
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_size()’ gets the size (number of elements) of a vector.
 */
int mtxvector_size(
    const struct mtxvector * x,
    int64_t * size)
{
    if (x->type == mtxbasevector) {
        *size = mtxbasevector_size(&x->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        *size = mtxblasvector_size(&x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        *size = mtxnullvector_size(&x->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        *size = mtxompvector_size(&x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_num_nonzeros()’ gets the number of explicitly
 * stored vector entries.
 */
int mtxvector_num_nonzeros(
    const struct mtxvector * x,
    int64_t * num_nonzeros)
{
    if (x->type == mtxbasevector) {
        *num_nonzeros = mtxbasevector_num_nonzeros(&x->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        *num_nonzeros = mtxblasvector_num_nonzeros(&x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        *num_nonzeros = mtxnullvector_num_nonzeros(&x->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        *num_nonzeros = mtxompvector_num_nonzeros(&x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_idx()’ gets a pointer to an array containing the offset
 * of each nonzero vector entry for a vector in packed storage format.
 */
int mtxvector_idx(
    const struct mtxvector * x,
    int64_t ** idx)
{
    if (x->type == mtxbasevector) {
        *idx = mtxbasevector_idx(&x->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        *idx = mtxblasvector_idx(&x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        *idx = mtxnullvector_idx(&x->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        *idx = mtxompvector_idx(&x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
    return MTX_SUCCESS;
}

/*
 * memory management
 */

/**
 * ‘mtxvector_free()’ frees storage allocated for a vector.
 */
void mtxvector_free(
    struct mtxvector * x)
{
    if (x->type == mtxbasevector) {
        mtxbasevector_free(&x->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        mtxblasvector_free(&x->storage.blas);
#endif
    } else if (x->type == mtxnullvector) {
        mtxnullvector_free(&x->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        mtxompvector_free(&x->storage.omp);
#endif
    }
}

/**
 * ‘mtxvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_alloc_copy(
    struct mtxvector * dst,
    const struct mtxvector * src)
{
    if (src->type == mtxbasevector) {
        dst->type = mtxbasevector;
        return mtxbasevector_alloc_copy(&dst->storage.base, &src->storage.base);
    } else if (src->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        dst->type = mtxblasvector;
        return mtxblasvector_alloc_copy(&dst->storage.blas, &src->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxnullvector) {
        dst->type = mtxnullvector;
        return mtxnullvector_alloc_copy(&dst->storage.null, &src->storage.null);
    } else if (src->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        dst->type = mtxompvector;
        return mtxompvector_alloc_copy(&dst->storage.omp, &src->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_init_copy(
    struct mtxvector * dst,
    const struct mtxvector * src)
{
    if (src->type == mtxbasevector) {
        dst->type = mtxbasevector;
        return mtxbasevector_init_copy(&dst->storage.base, &src->storage.base);
    } else if (src->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        dst->type = mtxblasvector;
        return mtxblasvector_init_copy(&dst->storage.blas, &src->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxnullvector) {
        dst->type = mtxnullvector;
        return mtxnullvector_init_copy(&dst->storage.null, &src->storage.null);
    } else if (src->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        dst->type = mtxompvector;
        return mtxompvector_init_copy(
            &dst->storage.omp, &src->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Dense vectors
 */

/**
 * ‘mtxvector_alloc()’ allocates a vector of the given type.
 */
int mtxvector_alloc(
    struct mtxvector * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_alloc(&x->storage.base, field, precision, size);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_alloc(&x->storage.blas, field, precision, size);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_alloc(&x->storage.null, field, precision, size);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_alloc(&x->storage.omp, field, precision, size);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_real_single()’ allocates and initialises a vector
 * with real, single precision coefficients.
 */
int mtxvector_init_real_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    const float * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_real_single(&x->storage.base, size, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_real_single(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_real_single(&x->storage.null, size, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_real_single(&x->storage.omp, size, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_real_double()’ allocates and initialises a vector
 * with real, double precision coefficients.
 */
int mtxvector_init_real_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    const double * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_real_double(&x->storage.base, size, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_real_double(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_real_double(&x->storage.null, size, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_real_double(&x->storage.omp, size, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_init_complex_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    const float (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_complex_single(&x->storage.base, size, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_complex_single(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_complex_single(&x->storage.null, size, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_complex_single(&x->storage.omp, size, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_init_complex_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    const double (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_complex_double(&x->storage.base, size, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_complex_double(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_complex_double(&x->storage.null, size, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_complex_double(&x->storage.omp, size, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_init_integer_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    const int32_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_integer_single(&x->storage.base, size, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_integer_single(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_integer_single(&x->storage.null, size, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_integer_single(&x->storage.omp, size, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_init_integer_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    const int64_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_integer_double(&x->storage.base, size, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_integer_double(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_integer_double(&x->storage.null, size, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_integer_double(&x->storage.omp, size, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_pattern()’ allocates and initialises a vector of
 * ones.
 */
int mtxvector_init_pattern(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_pattern(&x->storage.base, size);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_pattern(&x->storage.blas, size);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_pattern(&x->storage.null, size);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_pattern(&x->storage.omp, size);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_strided_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_init_strided_real_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int stride,
    const float * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_strided_real_single(&x->storage.base, size, stride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_strided_real_single(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_strided_real_single(&x->storage.null, size, stride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_strided_real_single(&x->storage.omp, size, stride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_strided_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_init_strided_real_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int stride,
    const double * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_strided_real_double(&x->storage.base, size, stride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_strided_real_double(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_strided_real_double(&x->storage.null, size, stride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_strided_real_double(&x->storage.omp, size, stride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_strided_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxvector_init_strided_complex_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int stride,
    const float (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_strided_complex_single(&x->storage.base, size, stride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_strided_complex_single(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_strided_complex_single(&x->storage.null, size, stride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_strided_complex_single(&x->storage.omp, size, stride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_strided_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxvector_init_strided_complex_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int stride,
    const double (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_strided_complex_double(&x->storage.base, size, stride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_strided_complex_double(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_strided_complex_double(&x->storage.null, size, stride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_strided_complex_double(&x->storage.omp, size, stride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_strided_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxvector_init_strided_integer_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int stride,
    const int32_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_strided_integer_single(&x->storage.base, size, stride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_strided_integer_single(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_strided_integer_single(&x->storage.null, size, stride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_strided_integer_single(&x->storage.omp, size, stride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_strided_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxvector_init_strided_integer_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int stride,
    const int64_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_strided_integer_double(&x->storage.base, size, stride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_strided_integer_double(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_strided_integer_double(&x->storage.null, size, stride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_strided_integer_double(&x->storage.omp, size, stride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxvector_alloc_packed()’ allocates a vector in packed form.
 */
int mtxvector_alloc_packed(
    struct mtxvector * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_alloc_packed(&x->storage.base, field, precision, size, num_nonzeros, idx);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_alloc_packed(&x->storage.blas, field, precision, size, num_nonzeros, idx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_alloc_packed(&x->storage.null, field, precision, size, num_nonzeros, idx);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_alloc_packed(&x->storage.omp, field, precision, size, num_nonzeros, idx);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_init_packed_real_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_real_single(&x->storage.base, size, num_nonzeros, idx, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_real_single(&x->storage.blas, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_real_single(&x->storage.null, size, num_nonzeros, idx, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_real_single(&x->storage.omp, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_init_packed_real_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_real_double(&x->storage.base, size, num_nonzeros, idx, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_real_double(&x->storage.blas, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_real_double(&x->storage.null, size, num_nonzeros, idx, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_real_double(&x->storage.omp, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxvector_init_packed_complex_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_complex_single(&x->storage.base, size, num_nonzeros, idx, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_complex_single(&x->storage.blas, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_complex_single(&x->storage.null, size, num_nonzeros, idx, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_complex_single(&x->storage.omp, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxvector_init_packed_complex_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_complex_double(&x->storage.base, size, num_nonzeros, idx, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_complex_double(&x->storage.blas, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_complex_double(&x->storage.null, size, num_nonzeros, idx, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_complex_double(&x->storage.omp, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxvector_init_packed_integer_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_integer_single(&x->storage.base, size, num_nonzeros, idx, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_integer_single(&x->storage.blas, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_integer_single(&x->storage.null, size, num_nonzeros, idx, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_integer_single(&x->storage.omp, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxvector_init_packed_integer_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_integer_double(&x->storage.base, size, num_nonzeros, idx, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_integer_double(&x->storage.blas, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_integer_double(&x->storage.null, size, num_nonzeros, idx, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_integer_double(&x->storage.omp, size, num_nonzeros, idx, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxvector_init_packed_pattern(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_pattern(&x->storage.base, size, num_nonzeros, idx);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_pattern(&x->storage.blas, size, num_nonzeros, idx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_pattern(&x->storage.null, size, num_nonzeros, idx);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_pattern(&x->storage.omp, size, num_nonzeros, idx);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxvector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxvector_alloc_packed_strided(
    struct mtxvector * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_alloc_packed_strided(&x->storage.base, field, precision, size, num_nonzeros, idxstride, idxbase, idx);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_alloc_packed_strided(&x->storage.blas, field, precision, size, num_nonzeros, idxstride, idxbase, idx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_alloc_packed_strided(&x->storage.null, field, precision, size, num_nonzeros, idxstride, idxbase, idx);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_alloc_packed_strided(&x->storage.omp, field, precision, size, num_nonzeros, idxstride, idxbase, idx);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_init_packed_strided_real_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_real_single(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_real_single(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_real_single(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_real_single(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_init_packed_strided_real_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_real_double(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_real_double(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_real_double(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_real_double(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_init_packed_strided_complex_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_complex_single(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_complex_single(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_complex_single(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_complex_single(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_init_packed_strided_complex_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2])
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_complex_double(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_complex_double(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_complex_double(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_complex_double(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_init_packed_strided_integer_single(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_integer_single(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_integer_single(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_integer_single(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_integer_single(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_init_packed_strided_integer_double(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_integer_double(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_integer_double(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_integer_double(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_integer_double(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxvector_init_packed_strided_pattern(
    struct mtxvector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_init_packed_strided_pattern(&x->storage.base, size, num_nonzeros, idxstride, idxbase, idx);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_init_packed_strided_pattern(&x->storage.blas, size, num_nonzeros, idxstride, idxbase, idx);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_init_packed_strided_pattern(&x->storage.null, size, num_nonzeros, idxstride, idxbase, idx);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_init_packed_strided_pattern(&x->storage.omp, size, num_nonzeros, idxstride, idxbase, idx);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * accessing values
 */

/**
 * ‘mtxvector_get_real_single()’ obtains the values of a vector of
 * single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_get_real_single(
    const struct mtxvector * x,
    int64_t size,
    int stride,
    float * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_get_real_single(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_get_real_single(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_get_real_single(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_get_real_single(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_get_real_double()’ obtains the values of a vector of
 * double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_get_real_double(
    const struct mtxvector * x,
    int64_t size,
    int stride,
    double * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_get_real_double(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_get_real_double(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_get_real_double(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_get_real_double(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_get_complex_single()’ obtains the values of a vector of
 * single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_get_complex_single(
    struct mtxvector * x,
    int64_t size,
    int stride,
    float (* a)[2])
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_get_complex_single(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_get_complex_single(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_get_complex_single(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_get_complex_single(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_get_complex_double()’ obtains the values of a vector of
 * double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_get_complex_double(
    struct mtxvector * x,
    int64_t size,
    int stride,
    double (* a)[2])
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_get_complex_double(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_get_complex_double(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_get_complex_double(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_get_complex_double(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_get_integer_single()’ obtains the values of a vector of
 * single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_get_integer_single(
    struct mtxvector * x,
    int64_t size,
    int stride,
    int32_t * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_get_integer_single(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_get_integer_single(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_get_integer_single(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_get_integer_single(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_get_integer_double()’ obtains the values of a vector of
 * double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_get_integer_double(
    struct mtxvector * x,
    int64_t size,
    int stride,
    int64_t * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_get_integer_double(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_get_integer_double(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_get_integer_double(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_get_integer_double(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Modifying values
 */

/**
 * ‘mtxvector_setzero()’ sets every value of a vector to zero.
 */
int mtxvector_setzero(
    struct mtxvector * x)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_setzero(&x->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_setzero(&x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_setzero(&x->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_setzero(&x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_real_single()’ sets every (nonzero) value
 * of a vector equal to a constant, single precision floating point
 * number.
 */
int mtxvector_set_constant_real_single(
    struct mtxvector * x, float a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_constant_real_single(&x->storage.base, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_constant_real_single(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_constant_real_single(&x->storage.null, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_constant_real_single(
            &x->storage.omp, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_real_double()’ sets every (nonzero) value
 * of a vector equal to a constant, double precision floating point
 * number.
 */
int mtxvector_set_constant_real_double(
    struct mtxvector * x, double a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_constant_real_double(&x->storage.base, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_constant_real_double(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_constant_real_double(&x->storage.null, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_constant_real_double(
            &x->storage.omp, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_complex_single()’ sets every (nonzero)
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxvector_set_constant_complex_single(
    struct mtxvector * x, float a[2])
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_constant_complex_single(&x->storage.base, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_constant_complex_single(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_constant_complex_single(&x->storage.null, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_constant_complex_single(
            &x->storage.omp, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_complex_double()’ sets every (nonzero)
 * value of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxvector_set_constant_complex_double(
    struct mtxvector * x, double a[2])
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_constant_complex_double(&x->storage.base, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_constant_complex_double(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_constant_complex_double(&x->storage.null, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_constant_complex_double(
            &x->storage.omp, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_integer_single()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_single(
    struct mtxvector * x, int32_t a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_constant_integer_single(&x->storage.base, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_constant_integer_single(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_constant_integer_single(&x->storage.null, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_constant_integer_single(
            &x->storage.omp, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_integer_double()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_double(
    struct mtxvector * x, int64_t a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_constant_integer_double(&x->storage.base, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_constant_integer_double(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_constant_integer_double(&x->storage.null, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_constant_integer_double(
            &x->storage.omp, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_real_single()’ sets values of a vector from an array
 * of single precision floating point numbers.
 */
int mtxvector_set_real_single(
    struct mtxvector * x,
    int64_t size,
    int stride,
    const float * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_real_single(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_real_single(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_real_single(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_real_single(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_real_double()’ sets values of a vector from an array
 * of double precision floating point numbers.
 */
int mtxvector_set_real_double(
    struct mtxvector * x,
    int64_t size,
    int stride,
    const double * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_real_double(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_real_double(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_real_double(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_real_double(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_complex_single()’ sets values of a vector from an
 * array of single precision floating point complex numbers.
 */
int mtxvector_set_complex_single(
    struct mtxvector * x,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_complex_single(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_complex_single(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_complex_single(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_complex_single(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_complex_double()’ sets values of a vector from an
 * array of double precision floating point complex numbers.
 */
int mtxvector_set_complex_double(
    struct mtxvector * x,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_complex_double(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_complex_double(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_complex_double(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_complex_double(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_integer_single()’ sets values of a vector from an
 * array of integers.
 */
int mtxvector_set_integer_single(
    struct mtxvector * x,
    int64_t size,
    int stride,
    const int32_t * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_integer_single(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_integer_single(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_integer_single(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_integer_single(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_integer_double()’ sets values of a vector from an
 * array of integers.
 */
int mtxvector_set_integer_double(
    struct mtxvector * x,
    int64_t size,
    int stride,
    const int64_t * a)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_set_integer_double(&x->storage.base, size, stride, a);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_set_integer_double(&x->storage.blas, size, stride, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_set_integer_double(&x->storage.null, size, stride, a);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_set_integer_double(&x->storage.omp, size, stride, a);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_from_mtxfile()’ converts to a vector from Matrix Market
 * format.
 */
int mtxvector_from_mtxfile(
    struct mtxvector * x,
    const struct mtxfile * mtxfile,
    enum mtxvectortype type)
{
    if (type == mtxbasevector) {
        x->type = mtxbasevector;
        return mtxbasevector_from_mtxfile(&x->storage.base, mtxfile);
    } else if (type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxblasvector;
        return mtxblasvector_from_mtxfile(&x->storage.blas, mtxfile);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxnullvector) {
        x->type = mtxnullvector;
        return mtxnullvector_from_mtxfile(&x->storage.null, mtxfile);
    } else if (type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxompvector;
        return mtxompvector_from_mtxfile(
            &x->storage.omp, mtxfile);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_to_mtxfile()’ converts a vector to Matrix Market format.
 */
int mtxvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_to_mtxfile(mtxfile, &x->storage.base, num_rows, idx, mtxfmt);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_to_mtxfile(mtxfile, &x->storage.blas, num_rows, idx, mtxfmt);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_to_mtxfile(mtxfile, &x->storage.null, num_rows, idx, mtxfmt);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_to_mtxfile(mtxfile, &x->storage.omp, num_rows, idx, mtxfmt);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * I/O functions
 */

/**
 * ‘mtxvector_read()’ reads a vector from a Matrix Market file.  The
 * file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the vector.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_read(
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype type,
    const char * path,
    bool gzip,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_read(
        &mtxfile, precision, path, gzip, lines_read, bytes_read);
    if (err) return err;

    err = mtxvector_from_mtxfile(x, &mtxfile, type);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_fread()’ reads a vector from a stream in Matrix Market
 * format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the vector.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_fread(
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype type,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_fread(
        &mtxfile, precision, f, lines_read, bytes_read, line_max, linebuf);
    if (err) return err;

    err = mtxvector_from_mtxfile(x, &mtxfile, type);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxvector_gzread()’ reads a vector from a gzip-compressed stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * The ‘type’ argument specifies which format to use for representing
 * the vector.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_gzread(
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype type,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_gzread(
        &mtxfile, precision, f, lines_read, bytes_read, line_max, linebuf);
    if (err) return err;

    err = mtxvector_from_mtxfile(x, &mtxfile, type);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
#endif

/**
 * ‘mtxvector_write()’ writes a vector to a Matrix Market file. The
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
int mtxvector_write(
    const struct mtxvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(&mtxfile, x, num_rows, idx, mtxfmt);
    if (err) return err;
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
 * ‘mtxvector_fwrite()’ writes a vector to a stream.
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
int mtxvector_fwrite(
    const struct mtxvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(&mtxfile, x, num_rows, idx, mtxfmt);
    if (err) return err;

    err = mtxfile_fwrite(
        &mtxfile, f, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxvector_gzwrite()’ writes a vector to a gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxvector_gzwrite(
    const struct mtxvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(&mtxfile, x, num_rows, idx, mtxfmt);
    if (err) return err;

    err = mtxfile_gzwrite(
        &mtxfile, f, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
#endif

/*
 * Partitioning
 */

/**
 * ‘mtxvector_split()’ splits a vector into multiple vectors according
 * to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxvector’. If successful, then ‘dsts[p]’ points to
 * a vector consisting of elements from ‘src’ that belong to the ‘p’th
 * part, as designated by the ‘parts’ array.
 *
 * Finally, the argument ‘invperm’ may either be ‘NULL’, in which case
 * it is ignored, or it must point to an array of length ‘size’, which
 * is used to store the inverse permutation obtained from sorting the
 * vector elements in ascending order according to their assigned
 * parts. That is, ‘invperm[i]’ is the original position (before
 * sorting) of the vector element that now occupies the ‘i’th position
 * among the sorted elements.
 *
 * The caller is responsible for calling ‘mtxvector_free()’ to free
 * storage allocated for each vector in the ‘dsts’ array.
 */
int mtxvector_split(
    int num_parts,
    struct mtxvector ** dsts,
    const struct mtxvector * src,
    int64_t size,
    int * parts,
    int64_t * invperm)
{
    if (src->type == mtxbasevector) {
        struct mtxbasevector ** basedsts = malloc(
            num_parts * sizeof(struct mtxbasevector *));
        if (!basedsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxbasevector;
            basedsts[p] = &dsts[p]->storage.base;
        }
        int err = mtxbasevector_split(
            num_parts, basedsts, &src->storage.base, size, parts, invperm);
        free(basedsts);
        return err;
    } else if (src->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        struct mtxblasvector ** blasdsts = malloc(
            num_parts * sizeof(struct mtxblasvector *));
        if (!blasdsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxblasvector;
            blasdsts[p] = &dsts[p]->storage.blas;
        }
        int err = mtxblasvector_split(
            num_parts, blasdsts, &src->storage.blas, size, parts, invperm);
        free(blasdsts);
        return err;
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxnullvector) {
        struct mtxnullvector ** nulldsts = malloc(
            num_parts * sizeof(struct mtxnullvector *));
        if (!nulldsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxnullvector;
            nulldsts[p] = &dsts[p]->storage.null;
        }
        int err = mtxnullvector_split(
            num_parts, nulldsts, &src->storage.null, size, parts, invperm);
        free(nulldsts);
        return err;
    } else if (src->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        struct mtxompvector ** ompdsts = malloc(
            num_parts * sizeof(struct mtxompvector *));
        if (!ompdsts) return MTX_ERR_ERRNO;
        for (int p = 0; p < num_parts; p++) {
            dsts[p]->type = mtxompvector;
            ompdsts[p] = &dsts[p]->storage.omp;
        }
        int err = mtxompvector_split(
            num_parts, ompdsts, &src->storage.omp, size, parts, invperm);
        free(ompdsts);
        return err;
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 */
int mtxvector_swap(
    struct mtxvector * x,
    struct mtxvector * y)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_swap(&x->storage.base, &y->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_swap(&x->storage.blas, &y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_swap(&x->storage.null, &y->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_swap(&x->storage.omp, &y->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_copy()’ copies values of a vector, ‘y = x’.
 */
int mtxvector_copy(
    struct mtxvector * y,
    const struct mtxvector * x)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (y->type == mtxbasevector) {
        return mtxbasevector_copy(&y->storage.base, &x->storage.base);
    } else if (y->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_copy(&y->storage.blas, &x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxnullvector) {
        return mtxnullvector_copy(&y->storage.null, &x->storage.null);
    } else if (y->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_copy(&y->storage.omp, &x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_sscal()’ scales a vector by a single precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_sscal(
    float a,
    struct mtxvector * x,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_sscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_sscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_sscal(a, &x->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_sscal(a, &x->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_dscal()’ scales a vector by a double precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_dscal(
    double a,
    struct mtxvector * x,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_dscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_dscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_dscal(a, &x->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_dscal(a, &x->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_cscal()’ scales a vector by a complex, single precision
 * floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_cscal(
    float a[2],
    struct mtxvector * x,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_cscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_cscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_cscal(a, &x->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_cscal(a, &x->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_zscal()’ scales a vector by a complex, double precision
 * floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_zscal(
    double a[2],
    struct mtxvector * x,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_zscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_zscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_zscal(a, &x->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_zscal(a, &x->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_saxpy()’ adds a vector to another vector multiplied by a
 * single precision floating point value, ‘y = a*x + y’.
 */
int mtxvector_saxpy(
    float a,
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (y->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_saxpy(a, &x->storage.base, &y->storage.base, num_flops);
    } else if (y->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_saxpy(a, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_saxpy(a, &x->storage.null, &y->storage.null, num_flops);
    } else if (y->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_saxpy(a, &x->storage.omp, &y->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_daxpy()’ adds a vector to another vector multiplied by a
 * double precision floating point value, ‘y = a*x + y’.
 */
int mtxvector_daxpy(
    double a,
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (y->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_daxpy(a, &x->storage.base, &y->storage.base, num_flops);
    } else if (y->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_daxpy(a, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_daxpy(a, &x->storage.null, &y->storage.null, num_flops);
    } else if (y->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_daxpy(a, &x->storage.omp, &y->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxvector_saypx(
    float a,
    struct mtxvector * y,
    const struct mtxvector * x,
    int64_t * num_flops)
{
    if (y->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_saypx(a, &y->storage.base, &x->storage.base, num_flops);
    } else if (y->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_saypx(a, &y->storage.blas, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_saypx(a, &y->storage.null, &x->storage.null, num_flops);
    } else if (y->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_saypx(a, &y->storage.omp, &x->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxvector_daypx(
    double a,
    struct mtxvector * y,
    const struct mtxvector * x,
    int64_t * num_flops)
{
    if (y->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_daypx(a, &y->storage.base, &x->storage.base, num_flops);
    } else if (y->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_daypx(a, &y->storage.blas, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_daypx(a, &y->storage.null, &x->storage.null, num_flops);
    } else if (y->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_daypx(a, &y->storage.omp, &x->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 */
int mtxvector_sdot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_sdot(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_sdot(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_sdot(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_sdot(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 */
int mtxvector_ddot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_ddot(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_ddot(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_ddot(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_ddot(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxvector_cdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_cdotu(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_cdotu(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_cdotu(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_cdotu(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxvector_zdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_zdotu(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_zdotu(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_zdotu(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_zdotu(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 */
int mtxvector_cdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_cdotc(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_cdotc(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_cdotc(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_cdotc(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 */
int mtxvector_zdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxbasevector_zdotc(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_zdotc(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxnullvector_zdotc(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_zdotc(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_snrm2(
    const struct mtxvector * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_snrm2(&x->storage.base, nrm2, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_snrm2(&x->storage.blas, nrm2, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_snrm2(&x->storage.null, nrm2, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_snrm2(&x->storage.omp, nrm2, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_dnrm2(
    const struct mtxvector * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_dnrm2(&x->storage.base, nrm2, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_dnrm2(&x->storage.blas, nrm2, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_dnrm2(&x->storage.null, nrm2, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_dnrm2(&x->storage.omp, nrm2, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_sasum()’ computes the sum of absolute values (1-norm) of
 * a vector in single precision floating point.  If the vector is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxvector_sasum(
    const struct mtxvector * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_sasum(&x->storage.base, asum, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_sasum(&x->storage.blas, asum, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_sasum(&x->storage.null, asum, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_sasum(&x->storage.omp, asum, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_dasum()’ computes the sum of absolute values (1-norm) of
 * a vector in double precision floating point.  If the vector is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginary parts is computed.
 */
int mtxvector_dasum(
    const struct mtxvector * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_dasum(&x->storage.base, asum, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_dasum(&x->storage.blas, asum, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_dasum(&x->storage.null, asum, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_dasum(&x->storage.omp, asum, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_iamax()’ finds the index of the first element having the
 * maximum absolute value.  If the vector is complex-valued, then the
 * index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxvector_iamax(
    const struct mtxvector * x,
    int * iamax)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_iamax(&x->storage.base, iamax);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_iamax(&x->storage.blas, iamax);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_iamax(&x->storage.null, iamax);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_iamax(&x->storage.omp, iamax);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Level 1 Sparse BLAS operations.
 */

/**
 * ‘mtxvector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_ussdot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_ussdot(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_ussdot(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_ussdot(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_ussdot(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_usddot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_usddot(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_usddot(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_usddot(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_usddot(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_uscdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_uscdotu(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_uscdotu(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_uscdotu(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_uscdotu(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_uszdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_uszdotu(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_uszdotu(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_uszdotu(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_uszdotu(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_uscdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_uscdotc(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_uscdotc(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_uscdotc(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_uscdotc(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_uszdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_uszdotc(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_uszdotc(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_uszdotc(&x->storage.null, &y->storage.null, dot, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_uszdotc(&x->storage.omp, &y->storage.omp, dot, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_ussaxpy()’ performs a sparse vector update, multiplying
 * a sparse vector ‘x’ in packed form by a scalar ‘alpha’ and adding
 * the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_ussaxpy(
    float alpha,
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_ussaxpy(alpha, &x->storage.base, &y->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_ussaxpy(alpha, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_ussaxpy(alpha, &x->storage.null, &y->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_ussaxpy(alpha, &x->storage.omp, &y->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_usdaxpy()’ performs a sparse vector update, multiplying
 * a sparse vector ‘x’ in packed form by a scalar ‘alpha’ and adding
 * the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_usdaxpy(
    double alpha,
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_usdaxpy(alpha, &x->storage.base, &y->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_usdaxpy(alpha, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_usdaxpy(alpha, &x->storage.null, &y->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_usdaxpy(alpha, &x->storage.omp, &y->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_uscaxpy()’ performs a sparse vector update, multiplying
 * a sparse vector ‘x’ in packed form by a scalar ‘alpha’ and adding
 * the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_uscaxpy(
    float alpha[2],
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_uscaxpy(alpha, &x->storage.base, &y->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_uscaxpy(alpha, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_uscaxpy(alpha, &x->storage.null, &y->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_uscaxpy(alpha, &x->storage.omp, &y->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_uszaxpy()’ performs a sparse vector update, multiplying
 * a sparse vector ‘x’ in packed form by a scalar ‘alpha’ and adding
 * the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_uszaxpy(
    double alpha[2],
    const struct mtxvector * x,
    struct mtxvector * y,
    int64_t * num_flops)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_uszaxpy(alpha, &x->storage.base, &y->storage.base, num_flops);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_uszaxpy(alpha, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_uszaxpy(alpha, &x->storage.null, &y->storage.null, num_flops);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_uszaxpy(alpha, &x->storage.omp, &y->storage.omp, num_flops);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_usga()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are allowed.
 */
int mtxvector_usga(
    struct mtxvector * x,
    const struct mtxvector * y)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_usga(&x->storage.base, &y->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_usga(&x->storage.blas, &y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_usga(&x->storage.null, &y->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_usga(&x->storage.omp, &y->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_usgz()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form, while zeroing the values
 * of the source vector ‘y’ that were copied to ‘x’. Repeated indices
 * in the packed vector are allowed.
 */
int mtxvector_usgz(
    struct mtxvector * x,
    struct mtxvector * y)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxbasevector) {
        return mtxbasevector_usgz(&x->storage.base, &y->storage.base);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_usgz(&x->storage.blas, &y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_usgz(&x->storage.null, &y->storage.null);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_usgz(&x->storage.omp, &y->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_ussc(
    struct mtxvector * y,
    const struct mtxvector * x)
{
    if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (y->type == mtxbasevector) {
        return mtxbasevector_ussc(&y->storage.base, &x->storage.base);
    } else if (y->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_ussc(&y->storage.blas, &x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxnullvector) {
        return mtxnullvector_ussc(&y->storage.null, &x->storage.null);
    } else if (y->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_ussc(&y->storage.omp, &x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxvector_usscga()’ performs a combined scatter-gather operation
 * from a sparse vector ‘x’ in packed form into another sparse vector
 * ‘z’ in packed form. Repeated indices in the packed vector ‘x’ are
 * not allowed, otherwise the result is undefined. They are, however,
 * allowed in the packed vector ‘z’.
 */
int mtxvector_usscga(
    struct mtxvector * z,
    const struct mtxvector * x)
{
    if (x->type != z->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (z->type == mtxbasevector) {
        return mtxbasevector_usscga(&z->storage.base, &x->storage.base);
    } else if (z->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_usscga(&z->storage.blas, &x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (z->type == mtxnullvector) {
        return mtxnullvector_usscga(&z->storage.null, &x->storage.null);
    } else if (z->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_usscga(&z->storage.omp, &x->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_recv()’.
 */
int mtxvector_send(
    const struct mtxvector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_send(
            &x->storage.base, offset, count, recipient, tag, comm, mpierrcode);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_send(
            &x->storage.blas, offset, count, recipient, tag, comm, mpierrcode);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_send(
            &x->storage.null, offset, count, recipient, tag, comm, mpierrcode);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_send(
            &x->storage.omp, offset, count, recipient, tag, comm, mpierrcode);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_send()’.
 */
int mtxvector_recv(
    struct mtxvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_recv(
            &x->storage.base, offset, count, sender, tag, comm, status, mpierrcode);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_recv(
            &x->storage.blas, offset, count, sender, tag, comm, status, mpierrcode);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_recv(
            &x->storage.null, offset, count, sender, tag, comm, status, mpierrcode);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_recv(
            &x->storage.omp, offset, count, sender, tag, comm, status, mpierrcode);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_irecv()’ performs a non-blocking receive of a vector
 * from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_send()’.
 */
int mtxvector_irecv(
    struct mtxvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode)
{
    if (x->type == mtxbasevector) {
        return mtxbasevector_irecv(
            &x->storage.base, offset, count, sender, tag, comm, request, mpierrcode);
    } else if (x->type == mtxblasvector) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxblasvector_irecv(
            &x->storage.blas, offset, count, sender, tag, comm, request, mpierrcode);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxnullvector) {
        return mtxnullvector_irecv(
            &x->storage.null, offset, count, sender, tag, comm, request, mpierrcode);
    } else if (x->type == mtxompvector) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxompvector_irecv(
            &x->storage.omp, offset, count, sender, tag, comm, request, mpierrcode);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}
#endif
