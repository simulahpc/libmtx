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
 * Data structures and routines for dense vectors, where vector
 * operations do nothing. Note that this produces incorrect results,
 * so it is only useful for the purpose of eliminating vector
 * operations while debugging and carrying out performance
 * measurements.
 */

#ifndef LIBMTX_LINALG_NULL_VECTOR_H
#define LIBMTX_LINALG_NULL_VECTOR_H

#include <libmtx/libmtx-config.h>

#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/linalg/base/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxvector;

/**
 * ‘mtxnullvector’ represents a vector, where vector operations are
 * implemented to do nothing.
 *
 * This produces incorrect results, so it is only useful for the
 * purpose of eliminating vector operations while debugging and
 * carrying out performance measurements.
 */
struct mtxnullvector
{
    /**
     * ‘base’ is the underlying vector.
     */
    struct mtxbasevector base;
};

/*
 * vector properties
 */

/**
 * ‘mtxnullvector_field()’ gets the field of a vector.
 */
enum mtxfield mtxnullvector_field(const struct mtxnullvector * x);

/**
 * ‘mtxnullvector_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxnullvector_precision(const struct mtxnullvector * x);

/**
 * ‘mtxnullvector_size()’ gets the size of a vector.
 */
int64_t mtxnullvector_size(const struct mtxnullvector * x);

/**
 * ‘mtxnullvector_num_nonzeros()’ gets the number of explicitly
 * stored vector entries.
 */
int64_t mtxnullvector_num_nonzeros(const struct mtxnullvector * x);

/**
 * ‘mtxnullvector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxnullvector_idx(const struct mtxnullvector * x);

/*
 * memory management
 */

/**
 * ‘mtxnullvector_free()’ frees storage allocated for a vector.
 */
void mtxnullvector_free(
    struct mtxnullvector * x);

/**
 * ‘mtxnullvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxnullvector_alloc_copy(
    struct mtxnullvector * dst,
    const struct mtxnullvector * src);

/**
 * ‘mtxnullvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxnullvector_init_copy(
    struct mtxnullvector * dst,
    const struct mtxnullvector * src);

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxnullvector_alloc()’ allocates a vector.
 */
int mtxnullvector_alloc(
    struct mtxnullvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxnullvector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxnullvector_init_real_single(
    struct mtxnullvector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxnullvector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxnullvector_init_real_double(
    struct mtxnullvector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxnullvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxnullvector_init_complex_single(
    struct mtxnullvector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxnullvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxnullvector_init_complex_double(
    struct mtxnullvector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxnullvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxnullvector_init_integer_single(
    struct mtxnullvector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxnullvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxnullvector_init_integer_double(
    struct mtxnullvector * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxnullvector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxnullvector_init_pattern(
    struct mtxnullvector * x,
    int64_t size);

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxnullvector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxnullvector_init_strided_real_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t stride,
    const float * data);

/**
 * ‘mtxnullvector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxnullvector_init_strided_real_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t stride,
    const double * data);

/**
 * ‘mtxnullvector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxnullvector_init_strided_complex_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2]);

/**
 * ‘mtxnullvector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxnullvector_init_strided_complex_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2]);

/**
 * ‘mtxnullvector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxnullvector_init_strided_integer_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data);

/**
 * ‘mtxnullvector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxnullvector_init_strided_integer_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data);

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxnullvector_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxnullvector_alloc_packed(
    struct mtxnullvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/**
 * ‘mtxnullvector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxnullvector_init_packed_real_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data);

/**
 * ‘mtxnullvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxnullvector_init_packed_real_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data);

/**
 * ‘mtxnullvector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxnullvector_init_packed_complex_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2]);

/**
 * ‘mtxnullvector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxnullvector_init_packed_complex_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2]);

/**
 * ‘mtxnullvector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxnullvector_init_packed_integer_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data);

/**
 * ‘mtxnullvector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxnullvector_init_packed_integer_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data);

/**
 * ‘mtxnullvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxnullvector_init_packed_pattern(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxnullvector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxnullvector_alloc_packed_strided(
    struct mtxnullvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/**
 * ‘mtxnullvector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxnullvector_init_packed_strided_real_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data);

/**
 * ‘mtxnullvector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxnullvector_init_packed_strided_real_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data);

/**
 * ‘mtxnullvector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxnullvector_init_packed_strided_complex_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2]);

/**
 * ‘mtxnullvector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxnullvector_init_packed_strided_complex_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2]);

/**
 * ‘mtxnullvector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxnullvector_init_packed_strided_integer_single(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data);

/**
 * ‘mtxnullvector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxnullvector_init_packed_strided_integer_double(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data);

/**
 * ‘mtxnullvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxnullvector_init_packed_strided_pattern(
    struct mtxnullvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/*
 * accessing values
 */

/**
 * ‘mtxnullvector_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxnullvector_get_real_single(
    const struct mtxnullvector * x,
    int64_t size,
    int stride,
    float * a);

/**
 * ‘mtxnullvector_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxnullvector_get_real_double(
    const struct mtxnullvector * x,
    int64_t size,
    int stride,
    double * a);

/**
 * ‘mtxnullvector_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxnullvector_get_complex_single(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    float (* a)[2]);

/**
 * ‘mtxnullvector_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxnullvector_get_complex_double(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    double (* a)[2]);

/**
 * ‘mtxnullvector_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxnullvector_get_integer_single(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    int32_t * a);

/**
 * ‘mtxnullvector_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxnullvector_get_integer_double(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    int64_t * a);

/*
 * modifying values
 */

/**
 * ‘mtxnullvector_setzero()’ sets every value of a vector to zero.
 */
int mtxnullvector_setzero(
    struct mtxnullvector * x);

/**
 * ‘mtxnullvector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxnullvector_set_constant_real_single(
    struct mtxnullvector * x,
    float a);

/**
 * ‘mtxnullvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxnullvector_set_constant_real_double(
    struct mtxnullvector * x,
    double a);

/**
 * ‘mtxnullvector_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxnullvector_set_constant_complex_single(
    struct mtxnullvector * x,
    float a[2]);

/**
 * ‘mtxnullvector_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxnullvector_set_constant_complex_double(
    struct mtxnullvector * x,
    double a[2]);

/**
 * ‘mtxnullvector_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxnullvector_set_constant_integer_single(
    struct mtxnullvector * x,
    int32_t a);

/**
 * ‘mtxnullvector_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxnullvector_set_constant_integer_double(
    struct mtxnullvector * x,
    int64_t a);

/**
 * ‘mtxnullvector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxnullvector_set_real_single(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxnullvector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxnullvector_set_real_double(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxnullvector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxnullvector_set_complex_single(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxnullvector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxnullvector_set_complex_double(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxnullvector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxnullvector_set_integer_single(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxnullvector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxnullvector_set_integer_double(
    struct mtxnullvector * x,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxnullvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxnullvector_from_mtxfile(
    struct mtxnullvector * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxnullvector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxnullvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxnullvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxnullvector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxnullvector’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxnullvector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxnullvector_split(
    int num_parts,
    struct mtxnullvector ** dsts,
    const struct mtxnullvector * src,
    int64_t size,
    int * parts,
    int64_t * invperm);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxnullvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_swap(
    struct mtxnullvector * x,
    struct mtxnullvector * y);

/**
 * ‘mtxnullvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_copy(
    struct mtxnullvector * y,
    const struct mtxnullvector * x);

/**
 * ‘mtxnullvector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxnullvector_sscal(
    float a,
    struct mtxnullvector * x,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxnullvector_dscal(
    double a,
    struct mtxnullvector * x,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxnullvector_cscal(
    float a[2],
    struct mtxnullvector * x,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxnullvector_zscal(
    double a[2],
    struct mtxnullvector * x,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_saxpy(
    float a,
    const struct mtxnullvector * x,
    struct mtxnullvector * y,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_daxpy(
    double a,
    const struct mtxnullvector * x,
    struct mtxnullvector * y,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_saypx(
    float a,
    struct mtxnullvector * y,
    const struct mtxnullvector * x,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_daypx(
    double a,
    struct mtxnullvector * y,
    const struct mtxnullvector * x,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_sdot(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_ddot(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_cdotu(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_zdotu(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_cdotc(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxnullvector_zdotc(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxnullvector_snrm2(
    const struct mtxnullvector * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxnullvector_dnrm2(
    const struct mtxnullvector * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxnullvector_sasum(
    const struct mtxnullvector * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxnullvector_dasum(
    const struct mtxnullvector * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxnullvector_iamax(
    const struct mtxnullvector * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxnullvector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxnullvector_ussdot(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxnullvector_usddot(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxnullvector_uscdotu(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxnullvector_uszdotu(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxnullvector_uscdotc(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxnullvector_uszdotc(
    const struct mtxnullvector * x,
    const struct mtxnullvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxnullvector_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxnullvector_ussaxpy(
    float alpha,
    const struct mtxnullvector * x,
    struct mtxnullvector * y,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxnullvector_usdaxpy(
    double alpha,
    const struct mtxnullvector * x,
    struct mtxnullvector * y,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxnullvector_uscaxpy(
    float alpha[2],
    const struct mtxnullvector * x,
    struct mtxnullvector * y,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxnullvector_uszaxpy(
    double alpha[2],
    const struct mtxnullvector * x,
    struct mtxnullvector * y,
    int64_t * num_flops);

/**
 * ‘mtxnullvector_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxnullvector_usga(
    struct mtxnullvector * x,
    const struct mtxnullvector * y);

/**
 * ‘mtxnullvector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxnullvector_usgz(
    struct mtxnullvector * x,
    struct mtxnullvector * y);

/**
 * ‘mtxnullvector_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxnullvector_ussc(
    struct mtxnullvector * y,
    const struct mtxnullvector * x);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxnullvector_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxnullvector_usscga(
    struct mtxnullvector * z,
    const struct mtxnullvector * x);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxnullvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxnullvector_recv()’.
 */
int mtxnullvector_send(
    const struct mtxnullvector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxnullvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxnullvector_send()’.
 */
int mtxnullvector_recv(
    struct mtxnullvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode);

/**
 * ‘mtxnullvector_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxnullvector_send()’.
 */
int mtxnullvector_irecv(
    struct mtxnullvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode);
#endif

#endif
