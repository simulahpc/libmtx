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
 * Data structures and routines for dense vectors with vector
 * operations accelerated by an external BLAS library.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/blas/vector.h>
#include <libmtx/linalg/local/vector.h>

#include <cblas.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * vector properties
 */

/**
 * ‘mtxblasvector_field()’ gets the field of a vector.
 */
enum mtxfield mtxblasvector_field(const struct mtxblasvector * x)
{
    return mtxvector_base_field(&x->base);
}

/**
 * ‘mtxblasvector_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxblasvector_precision(const struct mtxblasvector * x)
{
    return mtxvector_base_precision(&x->base);
}

/**
 * ‘mtxblasvector_size()’ gets the size of a vector.
 */
int64_t mtxblasvector_size(const struct mtxblasvector * x)
{
    return mtxvector_base_size(&x->base);
}

/**
 * ‘mtxblasvector_num_nonzeros()’ gets the number of explicitly
 * stored vector entries.
 */
int64_t mtxblasvector_num_nonzeros(const struct mtxblasvector * x)
{
    return mtxvector_base_num_nonzeros(&x->base);
}

/**
 * ‘mtxblasvector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxblasvector_idx(const struct mtxblasvector * x)
{
    return mtxvector_base_idx(&x->base);
}

/*
 * memory management
 */

/**
 * ‘mtxblasvector_free()’ frees storage allocated for a vector.
 */
void mtxblasvector_free(
    struct mtxblasvector * x)
{
    mtxvector_base_free(&x->base);
}

/**
 * ‘mtxblasvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxblasvector_alloc_copy(
    struct mtxblasvector * dst,
    const struct mtxblasvector * src)
{
    return mtxvector_base_alloc_copy(&dst->base, &src->base);
}

/**
 * ‘mtxblasvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxblasvector_init_copy(
    struct mtxblasvector * dst,
    const struct mtxblasvector * src)
{
    return mtxvector_base_init_copy(&dst->base, &src->base);
}

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxblasvector_alloc()’ allocates a vector.
 */
int mtxblasvector_alloc(
    struct mtxblasvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    return mtxvector_base_alloc(&x->base, field, precision, size);
}

/**
 * ‘mtxblasvector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxblasvector_init_real_single(
    struct mtxblasvector * x,
    int64_t size,
    const float * data)
{
    return mtxvector_base_init_real_single(&x->base, size, data);
}

/**
 * ‘mtxblasvector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxblasvector_init_real_double(
    struct mtxblasvector * x,
    int64_t size,
    const double * data)
{
    return mtxvector_base_init_real_double(&x->base, size, data);
}

/**
 * ‘mtxblasvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxblasvector_init_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    const float (* data)[2])
{
    return mtxvector_base_init_complex_single(&x->base, size, data);
}

/**
 * ‘mtxblasvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxblasvector_init_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    const double (* data)[2])
{
    return mtxvector_base_init_complex_double(&x->base, size, data);
}

/**
 * ‘mtxblasvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxblasvector_init_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    const int32_t * data)
{
    return mtxvector_base_init_integer_single(&x->base, size, data);
}

/**
 * ‘mtxblasvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxblasvector_init_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    const int64_t * data)
{
    return mtxvector_base_init_integer_double(&x->base, size, data);
}

/**
 * ‘mtxblasvector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxblasvector_init_pattern(
    struct mtxblasvector * x,
    int64_t size)
{
    return mtxvector_base_init_pattern(&x->base, size);
}

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxblasvector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxblasvector_init_strided_real_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const float * data)
{
    return mtxvector_base_init_strided_real_single(&x->base, size, stride, data);
}

/**
 * ‘mtxblasvector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxblasvector_init_strided_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const double * data)
{
    return mtxvector_base_init_strided_real_double(&x->base, size, stride, data);
}

/**
 * ‘mtxblasvector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxblasvector_init_strided_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2])
{
    return mtxvector_base_init_strided_complex_single(&x->base, size, stride, data);
}

/**
 * ‘mtxblasvector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxblasvector_init_strided_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2])
{
    return mtxvector_base_init_strided_complex_double(&x->base, size, stride, data);
}

/**
 * ‘mtxblasvector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxblasvector_init_strided_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data)
{
    return mtxvector_base_init_strided_integer_single(&x->base, size, stride, data);
}

/**
 * ‘mtxblasvector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxblasvector_init_strided_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data)
{
    return mtxvector_base_init_strided_integer_double(&x->base, size, stride, data);
}

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxblasvector_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxblasvector_alloc_packed(
    struct mtxblasvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    return mtxvector_base_alloc_packed(&x->base, field, precision, size, num_nonzeros, idx);
}

/**
 * ‘mtxblasvector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxblasvector_init_packed_real_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data)
{
    return mtxvector_base_init_packed_real_single(&x->base, size, num_nonzeros, idx, data);
}

/**
 * ‘mtxblasvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxblasvector_init_packed_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data)
{
    return mtxvector_base_init_packed_real_double(&x->base, size, num_nonzeros, idx, data);
}

/**
 * ‘mtxblasvector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxblasvector_init_packed_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2])
{
    return mtxvector_base_init_packed_complex_single(&x->base, size, num_nonzeros, idx, data);
}

/**
 * ‘mtxblasvector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxblasvector_init_packed_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2])
{
    return mtxvector_base_init_packed_complex_double(&x->base, size, num_nonzeros, idx, data);
}

/**
 * ‘mtxblasvector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxblasvector_init_packed_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data)
{
    return mtxvector_base_init_packed_integer_single(&x->base, size, num_nonzeros, idx, data);
}

/**
 * ‘mtxblasvector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxblasvector_init_packed_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data)
{
    return mtxvector_base_init_packed_integer_double(&x->base, size, num_nonzeros, idx, data);
}

/**
 * ‘mtxblasvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxblasvector_init_packed_pattern(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    return mtxvector_base_init_packed_pattern(&x->base, size, num_nonzeros, idx);
}

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxblasvector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxblasvector_alloc_packed_strided(
    struct mtxblasvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    return mtxvector_base_alloc_packed_strided(&x->base, field, precision, size, num_nonzeros, idxstride, idxbase, idx);
}

/**
 * ‘mtxblasvector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxblasvector_init_packed_strided_real_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data)
{
    return mtxvector_base_init_packed_strided_real_single(&x->base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
}

/**
 * ‘mtxblasvector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxblasvector_init_packed_strided_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data)
{
    return mtxvector_base_init_packed_strided_real_double(&x->base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
}

/**
 * ‘mtxblasvector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxblasvector_init_packed_strided_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2])
{
    return mtxvector_base_init_packed_strided_complex_single(&x->base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
}

/**
 * ‘mtxblasvector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxblasvector_init_packed_strided_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2])
{
    return mtxvector_base_init_packed_strided_complex_double(&x->base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
}

/**
 * ‘mtxblasvector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxblasvector_init_packed_strided_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data)
{
    return mtxvector_base_init_packed_strided_integer_single(&x->base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
}

/**
 * ‘mtxblasvector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxblasvector_init_packed_strided_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data)
{
    return mtxvector_base_init_packed_strided_integer_double(&x->base, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
}

/**
 * ‘mtxblasvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxblasvector_init_packed_strided_pattern(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    return mtxvector_base_init_packed_strided_pattern(&x->base, size, num_nonzeros, idxstride, idxbase, idx);
}

/*
 * accessing values
 */

/**
 * ‘mtxblasvector_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxblasvector_get_real_single(
    const struct mtxblasvector * x,
    int64_t size,
    int stride,
    float * a)
{
    return mtxvector_base_get_real_single(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxblasvector_get_real_double(
    const struct mtxblasvector * x,
    int64_t size,
    int stride,
    double * a)
{
    return mtxvector_base_get_real_double(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxblasvector_get_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    float (* a)[2])
{
    return mtxvector_base_get_complex_single(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxblasvector_get_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    double (* a)[2])
{
    return mtxvector_base_get_complex_double(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxblasvector_get_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    int32_t * a)
{
    return mtxvector_base_get_integer_single(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxblasvector_get_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    int64_t * a)
{
    return mtxvector_base_get_integer_double(&x->base, size, stride, a);
}

/*
 * Modifying values
 */

/**
 * ‘mtxblasvector_setzero()’ sets every value of a vector to zero.
 */
int mtxblasvector_setzero(
    struct mtxblasvector * x)
{
    return mtxvector_base_setzero(&x->base);
}

/**
 * ‘mtxblasvector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxblasvector_set_constant_real_single(
    struct mtxblasvector * x,
    float a)
{
    return mtxvector_base_set_constant_real_single(&x->base, a);
}

/**
 * ‘mtxblasvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxblasvector_set_constant_real_double(
    struct mtxblasvector * x,
    double a)
{
    return mtxvector_base_set_constant_real_double(&x->base, a);
}

/**
 * ‘mtxblasvector_set_constant_complex_single()’ sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtxblasvector_set_constant_complex_single(
    struct mtxblasvector * x,
    float a[2])
{
    return mtxvector_base_set_constant_complex_single(&x->base, a);
}

/**
 * ‘mtxblasvector_set_constant_complex_double()’ sets every value of a
 * vector equal to a constant, double precision floating point complex
 * number.
 */
int mtxblasvector_set_constant_complex_double(
    struct mtxblasvector * x,
    double a[2])
{
    return mtxvector_base_set_constant_complex_double(&x->base, a);
}

/**
 * ‘mtxblasvector_set_constant_integer_single()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxblasvector_set_constant_integer_single(
    struct mtxblasvector * x,
    int32_t a)
{
    return mtxvector_base_set_constant_integer_single(&x->base, a);
}

/**
 * ‘mtxblasvector_set_constant_integer_double()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxblasvector_set_constant_integer_double(
    struct mtxblasvector * x,
    int64_t a)
{
    return mtxvector_base_set_constant_integer_double(&x->base, a);
}

/**
 * ‘mtxblasvector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxblasvector_set_real_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxvector_base_set_real_single(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxblasvector_set_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxvector_base_set_real_double(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxblasvector_set_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxvector_base_set_complex_single(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxblasvector_set_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxvector_base_set_complex_double(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxblasvector_set_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxvector_base_set_integer_single(&x->base, size, stride, a);
}

/**
 * ‘mtxblasvector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxblasvector_set_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxvector_base_set_integer_double(&x->base, size, stride, a);
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxblasvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxblasvector_from_mtxfile(
    struct mtxblasvector * x,
    const struct mtxfile * mtxfile)
{
    return mtxvector_base_from_mtxfile(&x->base, mtxfile);
}

/**
 * ‘mtxblasvector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxblasvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxblasvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    return mtxvector_base_to_mtxfile(mtxfile, &x->base, num_rows, idx, mtxfmt);
}

/*
 * Partitioning
 */

/**
 * ‘mtxblasvector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxblasvector’. If successful, then ‘dsts[p]’
 * points to a vector consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * Finally, the argument ‘invperm’ may either be ‘NULL’, in which case
 * it is ignored, or it must point to an array of length ‘size’, which
 * is used to store the inverse permutation obtained from sorting the
 * vector elements in ascending order according to their assigned
 * parts. That is, ‘invperm[i]’ is the original position (before
 * sorting) of the vector element that now occupies the ‘i’th position
 * among the sorted elements.
 *
 * The caller is responsible for calling ‘mtxblasvector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxblasvector_split(
    int num_parts,
    struct mtxblasvector ** dsts,
    const struct mtxblasvector * src,
    int64_t size,
    int * parts,
    int64_t * invperm)
{
    struct mtxvector_base ** basedsts = malloc(
        num_parts * sizeof(struct mtxvector_base *));
    if (!basedsts) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) basedsts[p] = &dsts[p]->base;
    int err = mtxvector_base_split(
        num_parts, basedsts, &src->base, size, parts, invperm);
    free(basedsts);
    return err;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxblasvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_swap(
    struct mtxblasvector * xblas,
    struct mtxblasvector * yblas)
{
#ifdef LIBMTX_HAVE_BLAS
    struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_sswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_dswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_cswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_zswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_swap(x, y);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_copy(
    struct mtxblasvector * yblas,
    const struct mtxblasvector * xblas)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_scopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_dcopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_ccopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_zcopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        return mtxvector_base_copy(y, x);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasvector_sscal(
    float a,
    struct mtxblasvector * xblas,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    struct mtxvector_base * x = &xblas->base;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            cblas_sscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            cblas_dscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            cblas_sscal(2*x->size, a, (float *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            cblas_dscal(2*x->size, a, (double *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_sscal(a, x, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasvector_dscal(
    double a,
    struct mtxblasvector * xblas,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    struct mtxvector_base * x = &xblas->base;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            cblas_sscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            cblas_dscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            cblas_sscal(2*x->size, a, (float *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            cblas_dscal(2*x->size, a, (double *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_dscal(a, x, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasvector_cscal(
    float a[2],
    struct mtxblasvector * xblas,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    struct mtxvector_base * x = &xblas->base;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        cblas_cscal(x->size, a, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        double az[2] = {a[0], a[1]};
        cblas_zscal(x->size, az, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasvector_zscal(
    double a[2],
    struct mtxblasvector * xblas,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    struct mtxvector_base * x = &xblas->base;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        float ac[2] = {a[0], a[1]};
        cblas_cscal(x->size, ac, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        cblas_zscal(x->size, a, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_saxpy(
    float a,
    const struct mtxblasvector * xblas,
    struct mtxblasvector * yblas,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_saxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_daxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_saxpy(2*y->size, a, (const float *) xdata, 1, (float *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_daxpy(2*y->size, a, (const double *) xdata, 1, (double *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        return mtxvector_base_saxpy(a, x, y, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_daxpy(
    double a,
    const struct mtxblasvector * xblas,
    struct mtxblasvector * yblas,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_saxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_daxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_saxpy(2*y->size, a, (const float *) xdata, 1, (float *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_daxpy(2*y->size, a, (const double *) xdata, 1, (double *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        return mtxvector_base_daxpy(a, x, y, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_caxpy(
    float a[2],
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxblasvector_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_zaxpy(
    double a[2],
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxblasvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_saypx(
    float a,
    struct mtxblasvector * yblas,
    const struct mtxblasvector * xblas,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    return mtxvector_base_saypx(a, y, x, num_flops);
}

/**
 * ‘mtxblasvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_daypx(
    double a,
    struct mtxblasvector * yblas,
    const struct mtxblasvector * xblas,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    return mtxvector_base_daypx(a, y, x, num_flops);
}

/**
 * ‘mtxblasvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_sdot(
    const struct mtxblasvector * xblas,
    const struct mtxblasvector * yblas,
    float * dot,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_sdot(x, y, dot, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_ddot(
    const struct mtxblasvector * xblas,
    const struct mtxblasvector * yblas,
    double * dot,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_ddot(x, y, dot, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_cdotu(
    const struct mtxblasvector * xblas,
    const struct mtxblasvector * yblas,
    float (* dot)[2],
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c[2] = {0, 0};
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxblasvector_sdot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_zdotu(
    const struct mtxblasvector * xblas,
    const struct mtxblasvector * yblas,
    double (* dot)[2],
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c[2] = {0, 0};
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxblasvector_ddot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_cdotc(
    const struct mtxblasvector * xblas,
    const struct mtxblasvector * yblas,
    float (* dot)[2],
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c[2] = {0, 0};
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxblasvector_sdot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_zdotc(
    const struct mtxblasvector * xblas,
    const struct mtxblasvector * yblas,
    double (* dot)[2],
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c[2] = {0, 0};
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxblasvector_ddot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxblasvector_snrm2(
    const struct mtxblasvector * xblas,
    float * nrm2,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *nrm2 = cblas_scnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *nrm2 = cblas_dznrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_snrm2(x, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxblasvector_dnrm2(
    const struct mtxblasvector * xblas,
    double * nrm2,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *nrm2 = cblas_scnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *nrm2 = cblas_dznrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_dnrm2(x, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasvector_sasum(
    const struct mtxblasvector * xblas,
    float * asum,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *asum = cblas_sasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *asum = cblas_dasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *asum = cblas_scasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *asum = cblas_dzasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_sasum(x, asum, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasvector_dasum(
    const struct mtxblasvector * xblas,
    double * asum,
    int64_t * num_flops)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *asum = cblas_sasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *asum = cblas_dasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *asum = cblas_scasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *asum = cblas_dzasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_dasum(x, asum, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/**
 * ‘mtxblasvector_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the vector is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxblasvector_iamax(
    const struct mtxblasvector * xblas,
    int * iamax)
{
#ifdef LIBMTX_HAVE_BLAS
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *iamax = cblas_isamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *iamax = cblas_idamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *iamax = cblas_icamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *iamax = cblas_izamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_iamax(x, iamax);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
#else
    return MTX_ERR_BLAS_NOT_SUPPORTED;
#endif
}

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxblasvector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxblasvector_ussdot(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    float * dot,
    int64_t * num_flops)
{
    return mtxvector_base_ussdot(&x->base, &y->base, dot, num_flops);
}

/**
 * ‘mtxblasvector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxblasvector_usddot(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    double * dot,
    int64_t * num_flops)
{
    return mtxvector_base_usddot(&x->base, &y->base, dot, num_flops);
}

/**
 * ‘mtxblasvector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxblasvector_uscdotu(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_base_uscdotu(&x->base, &y->base, dot, num_flops);
}

/**
 * ‘mtxblasvector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxblasvector_uszdotu(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_base_uszdotu(&x->base, &y->base, dot, num_flops);
}

/**
 * ‘mtxblasvector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxblasvector_uscdotc(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_base_uscdotc(&x->base, &y->base, dot, num_flops);
}

/**
 * ‘mtxblasvector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxblasvector_uszdotc(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    return mtxvector_base_uszdotc(&x->base, &y->base, dot, num_flops);
}

/**
 * ‘mtxblasvector_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxblasvector_ussaxpy(
    float alpha,
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops)
{
    return mtxvector_base_ussaxpy(alpha, &x->base, &y->base, num_flops);
}

/**
 * ‘mtxblasvector_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxblasvector_usdaxpy(
    double alpha,
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops)
{
    return mtxvector_base_usdaxpy(alpha, &x->base, &y->base, num_flops);
}

/**
 * ‘mtxblasvector_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxblasvector_uscaxpy(
    float alpha[2],
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops)
{
    return mtxvector_base_uscaxpy(alpha, &x->base, &y->base, num_flops);
}

/**
 * ‘mtxblasvector_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxblasvector_uszaxpy(
    double alpha[2],
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops)
{
    return mtxvector_base_uszaxpy(alpha, &x->base, &y->base, num_flops);
}

/**
 * ‘mtxblasvector_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxblasvector_usga(
    struct mtxblasvector * x,
    const struct mtxblasvector * y)
{
    return mtxvector_base_usga(&x->base, &y->base);
}

/**
 * ‘mtxblasvector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxblasvector_usgz(
    struct mtxblasvector * x,
    struct mtxblasvector * y)
{
    return mtxvector_base_usgz(&x->base, &y->base);
}

/**
 * ‘mtxblasvector_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxblasvector_ussc(
    struct mtxblasvector * y,
    const struct mtxblasvector * x)
{
    return mtxvector_base_ussc(&y->base, &x->base);
}

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxblasvector_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxblasvector_usscga(
    struct mtxblasvector * z,
    const struct mtxblasvector * x)
{
    return mtxvector_base_usscga(&z->base, &x->base);
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxblasvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxblasvector_recv()’.
 */
int mtxblasvector_send(
    const struct mtxblasvector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    return mtxvector_base_send(
        &x->base, offset, count, recipient, tag, comm, mpierrcode);
}

/**
 * ‘mtxblasvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxblasvector_send()’.
 */
int mtxblasvector_recv(
    struct mtxblasvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode)
{
    return mtxvector_base_recv(
        &x->base, offset, count, sender, tag, comm, status, mpierrcode);
}

/**
 * ‘mtxblasvector_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxblasvector_send()’.
 */
int mtxblasvector_irecv(
    struct mtxblasvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode)
{
    return mtxvector_base_irecv(
        &x->base, offset, count, sender, tag, comm, request, mpierrcode);
}
#endif
