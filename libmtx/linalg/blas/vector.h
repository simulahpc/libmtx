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

#ifndef LIBMTX_LINALG_BLAS_VECTOR_H
#define LIBMTX_LINALG_BLAS_VECTOR_H

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
 * ‘mtxblasvector’ represents a dense vector that can perform
 * operations accelerated by an external BLAS library.
 */
struct mtxblasvector
{
    /**
     * ‘base’ is the underlying dense vector.
     */
    struct mtxbasevector base;
};

/*
 * vector properties
 */

/**
 * ‘mtxblasvector_field()’ gets the field of a vector.
 */
enum mtxfield mtxblasvector_field(const struct mtxblasvector * x);

/**
 * ‘mtxblasvector_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxblasvector_precision(const struct mtxblasvector * x);

/**
 * ‘mtxblasvector_size()’ gets the size of a vector.
 */
int64_t mtxblasvector_size(const struct mtxblasvector * x);

/**
 * ‘mtxblasvector_num_nonzeros()’ gets the number of explicitly
 * stored vector entries.
 */
int64_t mtxblasvector_num_nonzeros(const struct mtxblasvector * x);

/**
 * ‘mtxblasvector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxblasvector_idx(const struct mtxblasvector * x);

/*
 * memory management
 */

/**
 * ‘mtxblasvector_free()’ frees storage allocated for a vector.
 */
void mtxblasvector_free(
    struct mtxblasvector * x);

/**
 * ‘mtxblasvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxblasvector_alloc_copy(
    struct mtxblasvector * dst,
    const struct mtxblasvector * src);

/**
 * ‘mtxblasvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxblasvector_init_copy(
    struct mtxblasvector * dst,
    const struct mtxblasvector * src);

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
    int64_t size);

/**
 * ‘mtxblasvector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxblasvector_init_real_single(
    struct mtxblasvector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxblasvector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxblasvector_init_real_double(
    struct mtxblasvector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxblasvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxblasvector_init_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxblasvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxblasvector_init_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxblasvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxblasvector_init_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxblasvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxblasvector_init_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxblasvector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxblasvector_init_pattern(
    struct mtxblasvector * x,
    int64_t size);

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
    const float * data);

/**
 * ‘mtxblasvector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxblasvector_init_strided_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const double * data);

/**
 * ‘mtxblasvector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxblasvector_init_strided_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2]);

/**
 * ‘mtxblasvector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxblasvector_init_strided_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2]);

/**
 * ‘mtxblasvector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxblasvector_init_strided_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data);

/**
 * ‘mtxblasvector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxblasvector_init_strided_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data);

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
    const int64_t * idx);

/**
 * ‘mtxblasvector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxblasvector_init_packed_real_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data);

/**
 * ‘mtxblasvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxblasvector_init_packed_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data);

/**
 * ‘mtxblasvector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxblasvector_init_packed_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2]);

/**
 * ‘mtxblasvector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxblasvector_init_packed_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2]);

/**
 * ‘mtxblasvector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxblasvector_init_packed_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data);

/**
 * ‘mtxblasvector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxblasvector_init_packed_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data);

/**
 * ‘mtxblasvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxblasvector_init_packed_pattern(
    struct mtxblasvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

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
    const int64_t * idx);

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
    const float * data);

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
    const double * data);

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
    const float (* data)[2]);

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
    const double (* data)[2]);

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
    const int32_t * data);

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
    const int64_t * data);

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
    const int64_t * idx);

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
    float * a);

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
    double * a);

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
    float (* a)[2]);

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
    double (* a)[2]);

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
    int32_t * a);

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
    int64_t * a);

/*
 * Modifying values
 */

/**
 * ‘mtxblasvector_setzero()’ sets every value of a vector to zero.
 */
int mtxblasvector_setzero(
    struct mtxblasvector * x);

/**
 * ‘mtxblasvector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxblasvector_set_constant_real_single(
    struct mtxblasvector * x,
    float a);

/**
 * ‘mtxblasvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxblasvector_set_constant_real_double(
    struct mtxblasvector * x,
    double a);

/**
 * ‘mtxblasvector_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxblasvector_set_constant_complex_single(
    struct mtxblasvector * x,
    float a[2]);

/**
 * ‘mtxblasvector_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxblasvector_set_constant_complex_double(
    struct mtxblasvector * x,
    double a[2]);

/**
 * ‘mtxblasvector_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxblasvector_set_constant_integer_single(
    struct mtxblasvector * x,
    int32_t a);

/**
 * ‘mtxblasvector_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxblasvector_set_constant_integer_double(
    struct mtxblasvector * x,
    int64_t a);

/**
 * ‘mtxblasvector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxblasvector_set_real_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxblasvector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxblasvector_set_real_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxblasvector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxblasvector_set_complex_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxblasvector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxblasvector_set_complex_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxblasvector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxblasvector_set_integer_single(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxblasvector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxblasvector_set_integer_double(
    struct mtxblasvector * x,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxblasvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxblasvector_from_mtxfile(
    struct mtxblasvector * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxblasvector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxblasvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxblasvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt);

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
    int64_t * invperm);

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
    struct mtxblasvector * x,
    struct mtxblasvector * y);

/**
 * ‘mtxblasvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_copy(
    struct mtxblasvector * y,
    const struct mtxblasvector * x);

/**
 * ‘mtxblasvector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasvector_sscal(
    float a,
    struct mtxblasvector * x,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxblasvector_dscal(
    double a,
    struct mtxblasvector * x,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasvector_cscal(
    float a[2],
    struct mtxblasvector * x,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxblasvector_zscal(
    double a[2],
    struct mtxblasvector * x,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_saxpy(
    float a,
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_daxpy(
    double a,
    const struct mtxblasvector * x,
    struct mtxblasvector * y,
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

/**
 * ‘mtxblasvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_saypx(
    float a,
    struct mtxblasvector * y,
    const struct mtxblasvector * x,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_daypx(
    double a,
    struct mtxblasvector * y,
    const struct mtxblasvector * x,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_sdot(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_ddot(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_cdotu(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_zdotu(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_cdotc(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxblasvector_zdotc(
    const struct mtxblasvector * x,
    const struct mtxblasvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxblasvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxblasvector_snrm2(
    const struct mtxblasvector * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxblasvector_dnrm2(
    const struct mtxblasvector * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasvector_sasum(
    const struct mtxblasvector * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxblasvector_dasum(
    const struct mtxblasvector * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxblasvector_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxblasvector_iamax(
    const struct mtxblasvector * x,
    int * iamax);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

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
    int64_t * num_flops);

/**
 * ‘mtxblasvector_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxblasvector_usga(
    struct mtxblasvector * x,
    const struct mtxblasvector * y);

/**
 * ‘mtxblasvector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxblasvector_usgz(
    struct mtxblasvector * x,
    struct mtxblasvector * y);

/**
 * ‘mtxblasvector_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxblasvector_ussc(
    struct mtxblasvector * y,
    const struct mtxblasvector * x);

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
    const struct mtxblasvector * x);

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
    int * mpierrcode);

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
    int * mpierrcode);

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
    int * mpierrcode);
#endif

#endif
