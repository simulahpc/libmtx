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
 * Last modified: 2022-04-26
 *
 * Data structures for vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
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
    case mtxvector_auto: return "auto";
    case mtxvector_array: return "array";
    case mtxvector_base: return "base";
    case mtxvector_blas: return "blas";
    case mtxvector_omp: return "omp";
    case mtxvector_coordinate: return "coordinate";
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
    if (strncmp("auto", t, strlen("auto")) == 0) {
        t += strlen("auto");
        *vector_type = mtxvector_auto;
    } else if (strncmp("array", t, strlen("array")) == 0) {
        t += strlen("array");
        *vector_type = mtxvector_array;
    } else if (strncmp("base", t, strlen("base")) == 0) {
        t += strlen("base");
        *vector_type = mtxvector_base;
    } else if (strncmp("blas", t, strlen("blas")) == 0) {
        t += strlen("blas");
        *vector_type = mtxvector_blas;
    } else if (strncmp("omp", t, strlen("omp")) == 0) {
        t += strlen("omp");
        *vector_type = mtxvector_omp;
    } else if (strncmp("coordinate", t, strlen("coordinate")) == 0) {
        t += strlen("coordinate");
        *vector_type = mtxvector_coordinate;
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_VECTOR_TYPE;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
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
    if (x->type == mtxvector_array) {
        mtxvector_array_free(&x->storage.array);
    } else if (x->type == mtxvector_base) {
        mtxvector_base_free(&x->storage.base);
    } else if (x->type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        mtxvector_blas_free(&x->storage.blas);
#endif
    } else if (x->type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        mtxvector_omp_free(&x->storage.omp);
#endif
    } else if (x->type == mtxvector_coordinate) {
        mtxvector_coordinate_free(&x->storage.coordinate);
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
    if (src->type == mtxvector_array) {
        dst->type = mtxvector_array;
        return mtxvector_array_alloc_copy(
            &dst->storage.array, &src->storage.array);
    } else if (src->type == mtxvector_base) {
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
    } else if (src->type == mtxvector_coordinate) {
        dst->type = mtxvector_coordinate;
        return mtxvector_coordinate_alloc_copy(
            &dst->storage.coordinate, &src->storage.coordinate);
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
    if (src->type == mtxvector_array) {
        dst->type = mtxvector_array;
        return mtxvector_array_init_copy(
            &dst->storage.array, &src->storage.array);
    } else if (src->type == mtxvector_array) {
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
    } else if (src->type == mtxvector_coordinate) {
        dst->type = mtxvector_coordinate;
        return mtxvector_coordinate_init_copy(
            &dst->storage.coordinate, &src->storage.coordinate);
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
    const float * data,
    int64_t stride)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_real_single(&x->storage.base, size, data, stride);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_real_single(&x->storage.blas, size, data, stride);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_real_single(&x->storage.omp, size, data, stride, 0);
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
    const double * data,
    int64_t stride)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_real_double(&x->storage.base, size, data, stride);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_real_double(&x->storage.blas, size, data, stride);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_real_double(&x->storage.omp, size, data, stride, 0);
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
    const float (* data)[2],
    int64_t stride)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_complex_single(&x->storage.base, size, data, stride);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_complex_single(&x->storage.blas, size, data, stride);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_complex_single(&x->storage.omp, size, data, stride, 0);
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
    const double (* data)[2],
    int64_t stride)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_complex_double(&x->storage.base, size, data, stride);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_complex_double(&x->storage.blas, size, data, stride);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_complex_double(&x->storage.omp, size, data, stride, 0);
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
    const int32_t * data,
    int64_t stride)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_integer_single(&x->storage.base, size, data, stride);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_integer_single(&x->storage.blas, size, data, stride);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_integer_single(&x->storage.omp, size, data, stride, 0);
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
    const int64_t * data,
    int64_t stride)
{
    if (type == mtxvector_base) {
        x->type = mtxvector_base;
        return mtxvector_base_init_strided_integer_double(&x->storage.base, size, data, stride);
    } else if (type == mtxvector_blas) {
#ifdef LIBMTX_HAVE_BLAS
        x->type = mtxvector_blas;
        return mtxvector_blas_init_strided_integer_double(&x->storage.blas, size, data, stride);
#else
        return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
    } else if (type == mtxvector_omp) {
#ifdef LIBMTX_HAVE_OPENMP
        x->type = mtxvector_omp;
        return mtxvector_omp_init_strided_integer_double(&x->storage.omp, size, data, stride, 0);
#else
        return MTX_ERR_OPENMP_NOT_SUPPORTED;
#endif
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Vector array formats
 */

/**
 * ‘mtxvector_alloc_array()’ allocates a vector in array
 * format.
 */
int mtxvector_alloc_array(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    x->type = mtxvector_array;
    return mtxvector_array_alloc(
        &x->storage.array, field, precision, size);
}

/**
 * ‘mtxvector_init_array_real_single()’ allocates and initialises a
 * vector in array format with real, single precision coefficients.
 */
int mtxvector_init_array_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data)
{
    x->type = mtxvector_array;
    return mtxvector_array_init_real_single(
        &x->storage.array, size, data);
}

/**
 * ‘mtxvector_init_array_real_double()’ allocates and initialises a
 * vector in array format with real, double precision coefficients.
 */
int mtxvector_init_array_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data)
{
    x->type = mtxvector_array;
    return mtxvector_array_init_real_double(
        &x->storage.array, size, data);
}

/**
 * ‘mtxvector_init_array_complex_single()’ allocates and initialises a
 * vector in array format with complex, single precision coefficients.
 */
int mtxvector_init_array_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2])
{
    x->type = mtxvector_array;
    return mtxvector_array_init_complex_single(
        &x->storage.array, size, data);
}

/**
 * ‘mtxvector_init_array_complex_double()’ allocates and initialises a
 * vector in array format with complex, double precision coefficients.
 */
int mtxvector_init_array_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2])
{
    x->type = mtxvector_array;
    return mtxvector_array_init_complex_double(
        &x->storage.array, size, data);
}

/**
 * ‘mtxvector_init_array_integer_single()’ allocates and initialises a
 * vector in array format with integer, single precision coefficients.
 */
int mtxvector_init_array_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data)
{
    x->type = mtxvector_array;
    return mtxvector_array_init_integer_single(
        &x->storage.array, size, data);
}

/**
 * ‘mtxvector_init_array_integer_double()’ allocates and initialises a
 * vector in array format with integer, double precision coefficients.
 */
int mtxvector_init_array_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data)
{
    x->type = mtxvector_array;
    return mtxvector_array_init_integer_double(
        &x->storage.array, size, data);
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

/**
 * ‘mtxvector_init_base_real_single()’ allocates and initialises a
 *  dense vector with real, single precision coefficients.
 */
int mtxvector_init_base_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data)
{
    x->type = mtxvector_base;
    return mtxvector_base_init_real_single(&x->storage.base, size, data);
}

/**
 * ‘mtxvector_init_base_real_double()’ allocates and initialises a
 *  dense vector with real, double precision coefficients.
 */
int mtxvector_init_base_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data)
{
    x->type = mtxvector_base;
    return mtxvector_base_init_real_double(&x->storage.base, size, data);
}

/**
 * ‘mtxvector_init_base_complex_single()’ allocates and initialises a
 *  dense vector with complex, single precision coefficients.
 */
int mtxvector_init_base_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2])
{
    x->type = mtxvector_base;
    return mtxvector_base_init_complex_single(&x->storage.base, size, data);
}

/**
 * ‘mtxvector_init_base_complex_double()’ allocates and initialises a
 *  dense vector with complex, double precision coefficients.
 */
int mtxvector_init_base_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2])
{
    x->type = mtxvector_base;
    return mtxvector_base_init_complex_double(&x->storage.base, size, data);
}

/**
 * ‘mtxvector_init_base_integer_single()’ allocates and initialises a
 *  dense vector with integer, single precision coefficients.
 */
int mtxvector_init_base_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data)
{
    x->type = mtxvector_base;
    return mtxvector_base_init_integer_single(&x->storage.base, size, data);
}

/**
 * ‘mtxvector_init_base_integer_double()’ allocates and initialises a
 *  dense vector with integer, double precision coefficients.
 */
int mtxvector_init_base_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data)
{
    x->type = mtxvector_base;
    return mtxvector_base_init_integer_double(&x->storage.base, size, data);
}

/*
 * Dense vectors with vector operations accelerated by an external
 * BLAS library.
 */

/**
 * ‘mtxvector_alloc_blas()’ allocates a dense vector.
 */
int mtxvector_alloc_blas(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_alloc(&x->storage.blas, field, precision, size);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_blas_real_single()’ allocates and initialises a
 *  dense vector with real, single precision coefficients.
 */
int mtxvector_init_blas_real_single(
    struct mtxvector * x,
    int64_t size,
    const float * data)
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_init_real_single(&x->storage.blas, size, data);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_blas_real_double()’ allocates and initialises a
 *  dense vector with real, double precision coefficients.
 */
int mtxvector_init_blas_real_double(
    struct mtxvector * x,
    int64_t size,
    const double * data)
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_init_real_double(&x->storage.blas, size, data);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_blas_complex_single()’ allocates and initialises a
 *  dense vector with complex, single precision coefficients.
 */
int mtxvector_init_blas_complex_single(
    struct mtxvector * x,
    int64_t size,
    const float (* data)[2])
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_init_complex_single(&x->storage.blas, size, data);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_blas_complex_double()’ allocates and initialises a
 *  dense vector with complex, double precision coefficients.
 */
int mtxvector_init_blas_complex_double(
    struct mtxvector * x,
    int64_t size,
    const double (* data)[2])
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_init_complex_double(&x->storage.blas, size, data);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_blas_integer_single()’ allocates and initialises a
 *  dense vector with integer, single precision coefficients.
 */
int mtxvector_init_blas_integer_single(
    struct mtxvector * x,
    int64_t size,
    const int32_t * data)
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_init_integer_single(&x->storage.blas, size, data);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxvector_init_blas_integer_double()’ allocates and initialises a
 *  dense vector with integer, double precision coefficients.
 */
int mtxvector_init_blas_integer_double(
    struct mtxvector * x,
    int64_t size,
    const int64_t * data)
{
#ifdef LIBMTX_HAVE_BLAS
    x->type = mtxvector_blas;
    return mtxvector_blas_init_integer_double(&x->storage.blas, size, data);
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
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
 * Vector coordinate formats
 */

/**
 * ‘mtxvector_alloc_coordinate()’ allocates a vector in
 * coordinate format.
 */
int mtxvector_alloc_coordinate(
    struct mtxvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros)
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_alloc(
        &x->storage.coordinate, field, precision, size, num_nonzeros);
}

/**
 * ‘mtxvector_init_coordinate_real_single()’ allocates and initialises
 * a vector in coordinate format with real, single precision
 * coefficients.
 */
int mtxvector_init_coordinate_real_single(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices,
    const float * data)
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_real_single(
        &x->storage.coordinate, size, num_nonzeros, indices, data);
}

/**
 * ‘mtxvector_init_coordinate_real_double()’ allocates and initialises
 * a vector in coordinate format with real, double precision
 * coefficients.
 */
int mtxvector_init_coordinate_real_double(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices,
    const double * data)
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_real_double(
        &x->storage.coordinate, size, num_nonzeros, indices, data);
}

/**
 * ‘mtxvector_init_coordinate_complex_single()’ allocates and
 * initialises a vector in coordinate format with complex, single
 * precision coefficients.
 */
int mtxvector_init_coordinate_complex_single(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices,
    const float (* data)[2])
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_complex_single(
        &x->storage.coordinate, size, num_nonzeros, indices, data);
}

/**
 * ‘mtxvector_init_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxvector_init_coordinate_complex_double(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices,
    const double (* data)[2])
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_complex_double(
        &x->storage.coordinate, size, num_nonzeros, indices, data);
}

/**
 * ‘mtxvector_init_coordinate_integer_single()’ allocates and
 * initialises a vector in coordinate format with integer, single
 * precision coefficients.
 */
int mtxvector_init_coordinate_integer_single(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices,
    const int32_t * data)
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_integer_single(
        &x->storage.coordinate, size, num_nonzeros, indices, data);
}

/**
 * ‘mtxvector_init_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxvector_init_coordinate_integer_double(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices,
    const int64_t * data)
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_integer_double(
        &x->storage.coordinate, size, num_nonzeros, indices, data);
}

/**
 * ‘mtxvector_init_coordinate_pattern()’ allocates and initialises a
 * vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxvector_init_coordinate_pattern(
    struct mtxvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int * indices)
{
    x->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_pattern(
        &x->storage.coordinate, size, num_nonzeros, indices);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_set_constant_real_single(
            &x->storage.array, a);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_real_single(
            &x->storage.coordinate, a);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_set_constant_real_double(
            &x->storage.array, a);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_real_double(
            &x->storage.coordinate, a);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_set_constant_complex_single(
            &x->storage.array, a);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_complex_single(
            &x->storage.coordinate, a);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_set_constant_complex_double(
            &x->storage.array, a);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_complex_double(
            &x->storage.coordinate, a);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_integer_single()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_single(
    struct mtxvector * x, int32_t a)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_set_constant_integer_single(
            &x->storage.array, a);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_integer_single(
            &x->storage.coordinate, a);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_set_constant_integer_double()’ sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_double(
    struct mtxvector * x, int64_t a)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_set_constant_integer_double(
            &x->storage.array, a);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_integer_double(
            &x->storage.coordinate, a);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_from_mtxfile(
    struct mtxvector * x,
    const struct mtxfile * mtxfile,
    enum mtxvectortype type)
{
    if (type == mtxvector_auto) {
        if (mtxfile->header.format == mtxfile_array) {
            type = mtxvector_array;
        } else if (mtxfile->header.format == mtxfile_coordinate) {
            type = mtxvector_coordinate;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    }

    if (type == mtxvector_array) {
        x->type = mtxvector_array;
        return mtxvector_array_from_mtxfile(
            &x->storage.array, mtxfile);
    } else if (type == mtxvector_base) {
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
    } else if (type == mtxvector_coordinate) {
        x->type = mtxvector_coordinate;
        return mtxvector_coordinate_from_mtxfile(
            &x->storage.coordinate, mtxfile);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_to_mtxfile()’ converts a vector to a vector in Matrix
 * Market format.
 */
int mtxvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_to_mtxfile(
            mtxfile, &x->storage.array, mtxfmt);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_to_mtxfile(
            mtxfile, &x->storage.coordinate, mtxfmt);
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
 * the vector.  If ‘type’ is ‘mtxvector_auto’, then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
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
 * the vector.  If ‘type’ is ‘mtxvector_auto’, then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
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
 * the vector.  If ‘type’ is ‘mtxvector_auto’, then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
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
    if (src->type == mtxvector_array) {
        return mtxvector_array_partition(
            dsts, &src->storage.array, part);
    } else if (src->type == mtxvector_base) {
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
    } else if (src->type == mtxvector_coordinate) {
        return mtxvector_coordinate_partition(
            dsts, &src->storage.coordinate, part);
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
    if (srcs[0].type == mtxvector_array) {
        dst->type = mtxvector_array;
        return mtxvector_array_join(
            &dst->storage.array, srcs, part);
    } else if (srcs[0].type == mtxvector_base) {
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
    } else if (srcs[0].type == mtxvector_coordinate) {
        dst->type = mtxvector_coordinate;
        return mtxvector_coordinate_join(
            &dst->storage.coordinate, srcs, part);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_swap(
            &x->storage.array, &y->storage.array);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_swap(
            &x->storage.coordinate, &y->storage.coordinate);
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
    if (y->type == mtxvector_array) {
        return mtxvector_array_copy(
            &y->storage.array, &x->storage.array);
    } else if (y->type == mtxvector_base) {
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
    } else if (y->type == mtxvector_coordinate) {
        return mtxvector_coordinate_copy(
            &y->storage.coordinate, &x->storage.coordinate);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_sscal(a, &x->storage.array, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sscal(a, &x->storage.coordinate, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_dscal(a, &x->storage.array, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_dscal(a, &x->storage.coordinate, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_cscal(a, &x->storage.array, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_cscal(a, &x->storage.coordinate, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_zscal(a, &x->storage.array, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_zscal(a, &x->storage.coordinate, num_flops);
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
    if (y->type == mtxvector_array) {
        return mtxvector_array_saxpy(a, x, &y->storage.array, num_flops);
    } else if (y->type == mtxvector_base) {
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
    } else if (y->type == mtxvector_coordinate) {
        return mtxvector_coordinate_saxpy(a, x, &y->storage.coordinate, num_flops);
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
    if (y->type == mtxvector_array) {
        return mtxvector_array_daxpy(a, x, &y->storage.array, num_flops);
    } else if (y->type == mtxvector_base) {
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
    } else if (y->type == mtxvector_coordinate) {
        return mtxvector_coordinate_daxpy(a, x, &y->storage.coordinate, num_flops);
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
    if (y->type == mtxvector_array) {
        return mtxvector_array_saypx(a, &y->storage.array, x, num_flops);
    } else if (y->type == mtxvector_base) {
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
    } else if (y->type == mtxvector_coordinate) {
        return mtxvector_coordinate_saypx(a, &y->storage.coordinate, x, num_flops);
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
    if (y->type == mtxvector_array) {
        return mtxvector_array_daypx(a, &y->storage.array, x, num_flops);
    } else if (y->type == mtxvector_base) {
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
    } else if (y->type == mtxvector_coordinate) {
        return mtxvector_coordinate_daypx(a, &y->storage.coordinate, x, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_sdot(&x->storage.array, y, dot, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sdot(&x->storage.coordinate, y, dot, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_ddot(&x->storage.array, y, dot, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_ddot(&x->storage.coordinate, y, dot, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_cdotu(&x->storage.array, y, dot, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_cdotu(&x->storage.coordinate, y, dot, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_zdotu(&x->storage.array, y, dot, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_zdotu(&x->storage.coordinate, y, dot, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_cdotc(&x->storage.array, y, dot, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_cdotc(&x->storage.coordinate, y, dot, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_zdotc(&x->storage.array, y, dot, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_zdotc(&x->storage.coordinate, y, dot, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_snrm2(&x->storage.array, nrm2, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_snrm2(&x->storage.coordinate, nrm2, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_dnrm2(&x->storage.array, nrm2, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_dnrm2(&x->storage.coordinate, nrm2, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_sasum(&x->storage.array, asum, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sasum(&x->storage.coordinate, asum, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_dasum(&x->storage.array, asum, num_flops);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_dasum(&x->storage.coordinate, asum, num_flops);
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
    if (x->type == mtxvector_array) {
        return mtxvector_array_iamax(&x->storage.array, iamax);
    } else if (x->type == mtxvector_base) {
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
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_iamax(&x->storage.coordinate, iamax);
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
 * ‘mtxvector_usga()’ performs a (sparse) gather from a vector ‘y’
 * into another vector ‘x’.
 */
int mtxvector_usga(
    const struct mtxvector * y,
    struct mtxvector * x)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_usga(y, &x->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_usga(y, &x->storage.coordinate);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_usga2()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are allowed.
 */
int mtxvector_usga2(
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
 * ‘mtxvector_usgz()’ performs a (sparse) gather from a vector ‘y’
 * into another vector ‘x’, while zeroing the corresponding elements
 * of ‘y’ that were copied to ‘x’.
 */
int mtxvector_usgz(
    const struct mtxvector * y,
    struct mtxvector * x);

/**
 * ‘mtxvector_ussc()’ performs a (sparse) scatter from a vector ‘x’
 * into another vector ‘y’.
 */
int mtxvector_ussc(
    const struct mtxvector * x,
    struct mtxvector * y)
{
    if (y->type == mtxvector_array) {
        return mtxvector_array_ussc(x, &y->storage.array);
    } else if (y->type == mtxvector_coordinate) {
        /* if (x->type != y->type) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE; */
        /* return mtxvector_coordinate_copy( */
        /*     &y->storage.coordinate, &x->storage.coordinate); */
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_ussc2()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_ussc2(
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
 * Sorting
 */

/**
 * ‘mtxvector_permute()’ permutes the elements of a vector according
 * to a given permutation.
 *
 * The array ‘perm’ should be an array of length ‘size’ that stores a
 * permutation of the integers ‘0,1,...,N-1’, where ‘N’ is the number
 * of vector elements.
 *
 * After permuting, the 1st vector element of the original vector is
 * now located at position ‘perm[0]’ in the sorted vector ‘x’, the 2nd
 * element is now at position ‘perm[1]’, and so on.
 */
int mtxvector_permute(
    struct mtxvector * x,
    int64_t offset,
    int64_t size,
    int64_t * perm)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_permute(&x->storage.array, offset, size, perm);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_permute(
            &x->storage.coordinate, offset, size, perm);
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}

/**
 * ‘mtxvector_sort()’ sorts elements of a vector by the given keys.
 *
 * The array ‘keys’ must be an array of length ‘size’ that stores a
 * 64-bit unsigned integer sorting key that is used to define the
 * order in which to sort the vector elements..
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’, which is then used to store the sorting permutation. That
 * is, ‘perm’ is a permutation of the integers ‘0,1,...,N-1’, where
 * ‘N’ is the number of vector elements, such that the 1st vector
 * element in the original vector is now located at position ‘perm[0]’
 * in the sorted vector ‘x’, the 2nd element is now at position
 * ‘perm[1]’, and so on.
 */
int mtxvector_sort(
    struct mtxvector * x,
    int64_t size,
    uint64_t * keys,
    int64_t * perm)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_sort(&x->storage.array, size, keys, perm);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sort(&x->storage.coordinate, size, keys, perm);
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

/**
 * ‘mtxvector_bcast()’ broadcasts a vector from an MPI root process to
 * other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_bcast()’.
 */
int mtxvector_bcast(
    struct mtxvector * x,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_gatherv()’ gathers a vector onto an MPI root process
 * from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_gatherv()’.
 */
int mtxvector_gatherv(
    const struct mtxvector * sendbuf,
    int64_t sendoffset,
    int sendcount,
    struct mtxvector * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_scatterv()’ scatters a vector from an MPI root process
 * to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_scatterv()’.
 */
int mtxvector_scatterv(
    const struct mtxvector * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    struct mtxvector * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_alltoallv()’ performs an all-to-all exchange of a vector
 * between MPI processes in a communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxvector_alltoallv()’.
 */
int mtxvector_alltoallv(
    const struct mtxvector * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    struct mtxvector * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (sendbuf->type == mtxvector_array && recvbuf->type == sendbuf->type) {
        return mtxvector_array_alltoallv(
            &sendbuf->storage.array, sendoffset, sendcounts, senddispls,
            &recvbuf->storage.array, recvoffset, recvcounts, recvdispls,
            comm, disterr);
    } else if (sendbuf->type == mtxvector_coordinate &&
               recvbuf->type == sendbuf->type)
    {
        return mtxvector_coordinate_alltoallv(
            &sendbuf->storage.coordinate, sendoffset, sendcounts, senddispls,
            &recvbuf->storage.coordinate, recvoffset, recvcounts, recvdispls,
            comm, disterr);
    } else if (sendbuf->type != recvbuf->type) {
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    } else { return MTX_ERR_INVALID_VECTOR_TYPE; }
}
#endif
