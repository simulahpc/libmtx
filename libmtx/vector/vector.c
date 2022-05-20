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
 * Last modified: 2022-05-20
 *
 * Data structures for vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/blas.h>
#include <libmtx/vector/omp.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
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
    case mtxvector_base: return "base";
    case mtxvector_blas: return "blas";
    case mtxvector_omp: return "omp";
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
        *vector_type = mtxvector_base;
    } else if (strncmp("blas", t, strlen("blas")) == 0) {
        t += strlen("blas");
        *vector_type = mtxvector_blas;
    } else if (strncmp("omp", t, strlen("omp")) == 0) {
        t += strlen("omp");
        *vector_type = mtxvector_omp;
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

/**
 * ‘mtxvector_field()’ gets the field of a vector.
 */
int mtxvector_field(
    const struct mtxvector * x,
    enum mtxfield * field)
{
    if (x->type == mtxvector_base) {
        *field = x->storage.base.field;
        return MTX_SUCCESS;
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        *field = x->storage.blas.base.field;
        return MTX_SUCCESS;
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
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
    if (x->type == mtxvector_base) {
        *precision = x->storage.base.precision;
        return MTX_SUCCESS;
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        *precision = x->storage.blas.base.precision;
        return MTX_SUCCESS;
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        *precision = x->storage.omp.base.precision;
        return MTX_SUCCESS;
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Memory management
 */

/**
 * ‘mtxvector_free()’ frees storage allocated for a vector.
 */
void mtxvector_free(
    struct mtxvector * x)
{
    if (x->type == mtxvector_base) {
        mtxvector_base_free(&x->storage.base);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        mtxvector_blas_free(&x->storage.blas);
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        mtxvector_omp_free(&x->storage.omp);
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
    if (src->type == mtxvector_base) {
        dst->type = mtxvector_base;
        return mtxvector_base_alloc_copy(&dst->storage.base, &src->storage.base);
    } else if (src->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        dst->type = mtxvector_blas;
        return mtxvector_blas_alloc_copy(&dst->storage.blas, &src->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        dst->type = mtxvector_omp;
        return mtxvector_omp_alloc_copy(&dst->storage.omp, &src->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * ‘mtxvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_init_copy(
    struct mtxvector * dst,
    const struct mtxvector * src)
{
    if (src->type == mtxvector_base) {
        dst->type = mtxvector_base;
        return mtxvector_base_init_copy(&dst->storage.base, &src->storage.base);
    } else if (src->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        dst->type = mtxvector_blas;
        return mtxvector_blas_init_copy(&dst->storage.blas, &src->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        dst->type = mtxvector_omp;
        return mtxvector_omp_init_copy(
            &dst->storage.omp, &src->storage.omp);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_alloc(&x->storage.base, field, precision, size);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_alloc(&x->storage.blas, field, precision, size);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_alloc(&x->storage.omp, field, precision, size, 0);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_real_single(&x->storage.base, size, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_real_single(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_real_single(&x->storage.omp, size, data, 0);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_real_double(&x->storage.base, size, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_real_double(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_real_double(&x->storage.omp, size, data, 0);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_complex_single(&x->storage.base, size, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_complex_single(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_complex_single(&x->storage.omp, size, data, 0);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_complex_double(&x->storage.base, size, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_complex_double(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_complex_double(&x->storage.omp, size, data, 0);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_integer_single(&x->storage.base, size, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_integer_single(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_integer_single(&x->storage.omp, size, data, 0);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_integer_double(&x->storage.base, size, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_integer_double(&x->storage.blas, size, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_integer_double(&x->storage.omp, size, data, 0);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_pattern(&x->storage.base, size);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_pattern(&x->storage.blas, size);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_pattern(&x->storage.omp, size, 0);
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
    int64_t stride,
    const float * data)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_real_single(&x->storage.base, size, stride, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_real_single(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_real_single(&x->storage.omp, size, stride, data, 0);
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
    int64_t stride,
    const double * data)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_real_double(&x->storage.base, size, stride, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_real_double(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_real_double(&x->storage.omp, size, stride, data, 0);
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
    int64_t stride,
    const float (* data)[2])
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_complex_single(&x->storage.base, size, stride, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_complex_single(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_complex_single(&x->storage.omp, size, stride, data, 0);
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
    int64_t stride,
    const double (* data)[2])
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_complex_double(&x->storage.base, size, stride, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_complex_double(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_complex_double(&x->storage.omp, size, stride, data, 0);
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
    int64_t stride,
    const int32_t * data)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_integer_single(&x->storage.base, size, stride, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_integer_single(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_integer_single(&x->storage.omp, size, stride, data, 0);
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
    int64_t stride,
    const int64_t * data)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_integer_double(&x->storage.base, size, stride, data);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_integer_double(&x->storage.blas, size, stride, data);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_integer_double(&x->storage.omp, size, stride, data, 0);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Basic, dense vectors
 */

/**
 * ‘mtxvector_alloc_base()’ allocates a dense vector.
 */
int mtxvector_alloc_base(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    x->type = mtxvector_base;
    return mtxvector_base_alloc(&x->storage.base, field, precision, size);
}

/*
 * OpenMP shared-memory parallel vectors
 */

/**
 * ‘mtxvector_alloc_omp()’ allocates an OpenMP shared-memory parallel
 * vector.
 */
int mtxvector_alloc_omp(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_alloc(
        &x->storage.omp, field, precision, size, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_omp_real_single()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with real, single precision
 *  coefficients.
 */
int mtxvector_init_omp_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data,
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_init_real_single(
        &x->storage.omp, size, data, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_omp_real_double()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with real, double precision
 *  coefficients.
 */
int mtxvector_init_omp_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data,
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_init_real_double(
        &x->storage.omp, size, data, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_omp_complex_single()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with complex, single
 *  precision coefficients.
 */
int mtxvector_init_omp_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2],
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_init_complex_single(
        &x->storage.omp, size, data, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_omp_complex_double()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with complex, double
 *  precision coefficients.
 */
int mtxvector_init_omp_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2],
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_init_complex_double(
        &x->storage.omp, size, data, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_omp_integer_single()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with integer, single
 *  precision coefficients.
 */
int mtxvector_init_omp_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data,
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_init_integer_single(
        &x->storage.omp, size, data, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_omp_integer_double()’ allocates and initialises an
 *  OpenMP shared-memory parallel vector with integer, double
 *  precision coefficients.
 */
int mtxvector_init_omp_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data,
    int num_threads)
{
#ifdef LIBMTX_HAVE_OPENMP
    x->type = mtxvector_omp;
    return mtxvector_omp_init_integer_double(
        &x->storage.omp, size, data, num_threads);
#else
    return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_setzero(&x->storage.base);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_setzero(&x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_setzero(&x->storage.omp);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_set_constant_real_single(&x->storage.base, a);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_set_constant_real_single(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_set_constant_real_single(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_set_constant_real_double(&x->storage.base, a);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_set_constant_real_double(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_set_constant_real_double(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_set_constant_complex_single(&x->storage.base, a);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_set_constant_complex_single(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_set_constant_complex_single(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_set_constant_complex_double(&x->storage.base, a);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_set_constant_complex_double(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_set_constant_complex_double(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_set_constant_integer_single(&x->storage.base, a);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_set_constant_integer_single(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_set_constant_integer_single(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_set_constant_integer_double(&x->storage.base, a);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_set_constant_integer_double(&x->storage.blas, a);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_set_constant_integer_double(
            &x->storage.omp, a);
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
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_from_mtxfile(&x->storage.base, mtxfile);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_from_mtxfile(&x->storage.blas, mtxfile);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_from_mtxfile(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_to_mtxfile(mtxfile, &x->storage.base, num_rows, idx, mtxfmt);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_to_mtxfile(mtxfile, &x->storage.blas, num_rows, idx, mtxfmt);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_to_mtxfile(mtxfile, &x->storage.omp, num_rows, idx, mtxfmt);
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
 * ‘mtxvector_partition()’ partitions a vector into blocks according
 * to the given partitioning.
 *
 * The partition ‘part’ is allowed to be ‘NULL’, in which case a
 * trivial, singleton partition is used to partition the entries of
 * the vector. Otherwise, ‘part’ must partition the entries of the
 * vector ‘src’. That is, ‘part->size’ must be equal to the size of
 * the vector.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P’ values of type ‘struct mtxvector’, where ‘P’ is the number of
 * parts, ‘part->num_parts’.
 *
 * The user is responsible for freeing storage allocated for each
 * vector in the ‘dsts’ array.
 */
int mtxvector_partition(
    struct mtxvector * dsts,
    const struct mtxvector * src,
    const struct mtxpartition * part)
{
    if (src->type == mtxvector_base) {
        return mtxvector_base_partition(dsts, &src->storage.base, part);
    } else if (src->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_partition(dsts, &src->storage.blas, part);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (src->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_partition(dsts, &src->storage.omp, part);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_join()’ joins together block vectors to form a larger
 * vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_join(
    struct mtxvector * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part)
{
    int num_parts = part ? part->num_parts : 1;
    if (num_parts <= 0)
        return MTX_SUCCESS;
    if (srcs[0].type == mtxvector_base) {
        dst->type = mtxvector_base;
        return mtxvector_base_join(
            &dst->storage.base, srcs, part);
    } else if (srcs[0].type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        dst->type = mtxvector_blas;
        return mtxvector_blas_join(&dst->storage.blas, srcs, part);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (srcs[0].type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        dst->type = mtxvector_omp;
        return mtxvector_omp_join(&dst->storage.omp, srcs, part);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_swap(&x->storage.base, &y->storage.base);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_swap(&x->storage.blas, &y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_swap(&x->storage.omp, &y->storage.omp);
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
    if (y->type == mtxvector_base) {
        return mtxvector_base_copy(&y->storage.base, &x->storage.base);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_copy(&y->storage.blas, &x->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_copy(&y->storage.omp, &x->storage.omp);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_sscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_sscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_sscal(a, &x->storage.omp, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_dscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_dscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_dscal(a, &x->storage.omp, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_cscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_cscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_cscal(a, &x->storage.omp, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_zscal(a, &x->storage.base, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_zscal(a, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_zscal(a, &x->storage.omp, num_flops);
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
    if (y->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_saxpy(a, &x->storage.base, &y->storage.base, num_flops);
    } else if (y->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_saxpy(a, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_saxpy(a, &x->storage.omp, &y->storage.omp, num_flops);
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
    if (y->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_daxpy(a, &x->storage.base, &y->storage.base, num_flops);
    } else if (y->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_daxpy(a, &x->storage.blas, &y->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_daxpy(a, &x->storage.omp, &y->storage.omp, num_flops);
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
    if (y->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_saypx(a, &y->storage.base, &x->storage.base, num_flops);
    } else if (y->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_saypx(a, &y->storage.blas, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_saypx(a, &y->storage.omp, &x->storage.omp, num_flops);
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
    if (y->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_daypx(a, &y->storage.base, &x->storage.base, num_flops);
    } else if (y->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_daypx(a, &y->storage.blas, &x->storage.blas, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_daypx(a, &y->storage.omp, &x->storage.omp, num_flops);
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
    if (x->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_sdot(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_sdot(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_sdot(&x->storage.omp, &y->storage.omp, dot, num_flops);
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
    if (x->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_ddot(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_ddot(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_ddot(&x->storage.omp, &y->storage.omp, dot, num_flops);
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
    if (x->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_cdotu(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_cdotu(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_cdotu(&x->storage.omp, &y->storage.omp, dot, num_flops);
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
    if (x->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_zdotu(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_zdotu(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_zdotu(&x->storage.omp, &y->storage.omp, dot, num_flops);
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
    if (x->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_cdotc(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_cdotc(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_cdotc(&x->storage.omp, &y->storage.omp, dot, num_flops);
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
    if (x->type == mtxvector_base) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
        return mtxvector_base_zdotc(&x->storage.base, &y->storage.base, dot, num_flops);
    } else if (x->type == mtxvector_blas) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_zdotc(&x->storage.blas, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
        if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_zdotc(&x->storage.omp, &y->storage.omp, dot, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_snrm2(&x->storage.base, nrm2, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_snrm2(&x->storage.blas, nrm2, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_snrm2(&x->storage.omp, nrm2, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_dnrm2(&x->storage.base, nrm2, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_dnrm2(&x->storage.blas, nrm2, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_dnrm2(&x->storage.omp, nrm2, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_sasum(&x->storage.base, asum, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_sasum(&x->storage.blas, asum, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_sasum(&x->storage.omp, asum, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_dasum(&x->storage.base, asum, num_flops);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_dasum(&x->storage.blas, asum, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_dasum(&x->storage.omp, asum, num_flops);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_iamax(&x->storage.base, iamax);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_iamax(&x->storage.blas, iamax);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_iamax(&x->storage.omp, iamax);
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
    const struct mtxvector_packed * x,
    const struct mtxvector * y,
    float * dot,
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_ussdot(x, &y->storage.base, dot, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_ussdot(x, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_ussdot(x, &y->storage.omp, dot, num_flops);
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
    const struct mtxvector_packed * x,
    const struct mtxvector * y,
    double * dot,
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_usddot(x, &y->storage.base, dot, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_usddot(x, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_usddot(x, &y->storage.omp, dot, num_flops);
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
    const struct mtxvector_packed * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_uscdotu(x, &y->storage.base, dot, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_uscdotu(x, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_uscdotu(x, &y->storage.omp, dot, num_flops);
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
    const struct mtxvector_packed * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_uszdotu(x, &y->storage.base, dot, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_uszdotu(x, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_uszdotu(x, &y->storage.omp, dot, num_flops);
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
    const struct mtxvector_packed * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_uscdotc(x, &y->storage.base, dot, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_uscdotc(x, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_uscdotc(x, &y->storage.omp, dot, num_flops);
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
    const struct mtxvector_packed * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_uszdotc(x, &y->storage.base, dot, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_uszdotc(x, &y->storage.blas, dot, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_uszdotc(x, &y->storage.omp, dot, num_flops);
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
    struct mtxvector * y,
    float alpha,
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_ussaxpy(&y->storage.base, alpha, x, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_ussaxpy(&y->storage.blas, alpha, x, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_ussaxpy(&y->storage.omp, alpha, x, num_flops);
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
    struct mtxvector * y,
    double alpha,
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_usdaxpy(&y->storage.base, alpha, x, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_usdaxpy(&y->storage.blas, alpha, x, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_usdaxpy(&y->storage.omp, alpha, x, num_flops);
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
    struct mtxvector * y,
    float alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_uscaxpy(&y->storage.base, alpha, x, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_uscaxpy(&y->storage.blas, alpha, x, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_uscaxpy(&y->storage.omp, alpha, x, num_flops);
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
    struct mtxvector * y,
    double alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_uszaxpy(&y->storage.base, alpha, x, num_flops);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_uszaxpy(&y->storage.blas, alpha, x, num_flops);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_uszaxpy(&y->storage.omp, alpha, x, num_flops);
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
    struct mtxvector_packed * x,
    const struct mtxvector * y)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_usga(x, &y->storage.base);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_usga(x, &y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_usga(x, &y->storage.omp);
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
    struct mtxvector_packed * x,
    struct mtxvector * y)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_usgz(x, &y->storage.base);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_usgz(x, &y->storage.blas);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_usgz(x, &y->storage.omp);
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
    const struct mtxvector_packed * x)
{
    if (y->type == mtxvector_base) {
        return mtxvector_base_ussc(&y->storage.base, x);
    } else if (y->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_ussc(&y->storage.blas, x);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (y->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_ussc(&y->storage.omp, x);
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
    struct mtxvector_packed * z,
    const struct mtxvector_packed * x)
{
    if (z->x.type == mtxvector_base) {
        return mtxvector_base_usscga(z, x);
    } else if (z->x.type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_usscga(z, x);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (z->x.type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_usscga(z, x);
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_send(
            &x->storage.base, offset, count, recipient, tag, comm, mpierrcode);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_send(
            &x->storage.blas, offset, count, recipient, tag, comm, mpierrcode);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_send(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_recv(
            &x->storage.base, offset, count, sender, tag, comm, status, mpierrcode);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_recv(
            &x->storage.blas, offset, count, sender, tag, comm, status, mpierrcode);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_recv(
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
    if (x->type == mtxvector_base) {
        return mtxvector_base_irecv(
            &x->storage.base, offset, count, sender, tag, comm, request, mpierrcode);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        return mtxvector_blas_irecv(
            &x->storage.blas, offset, count, sender, tag, comm, request, mpierrcode);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        return mtxvector_omp_irecv(
            &x->storage.omp, offset, count, sender, tag, comm, request, mpierrcode);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}
#endif
