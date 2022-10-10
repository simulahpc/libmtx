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
 * Last modified: 2022-10-09
 *
 * Data structures and routines for basic dense vectors.
 */

#ifndef LIBMTX_LINALG_BASE_VECTOR_H
#define LIBMTX_LINALG_BASE_VECTOR_H

#include <libmtx/libmtx-config.h>

#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxvector;

/**
 * ‘mtxbasevector’ represents a vector stored as a contiguous array
 * of elements in full or packed storage format.
 *
 * The vector is represented by a contiguous array of elements. If the
 * vector is stored in packed format, then there is also an array of
 * integers designating the offset of each element.
 */
struct mtxbasevector
{
    /**
     * ‘field’ is the vector field: ‘real’, ‘complex’, ‘integer’ or
     * ‘pattern’.
     */
    enum mtxfield field;

    /**
     * ‘precision’ is the precision used to store values.
     */
    enum mtxprecision precision;

    /**
     * ‘size’ is the number of vector elements.
     */
    int64_t size;

    /**
     * ‘num_nonzeros’ is the number of explicitly stored vector
     * entries. This must be equal to ‘size’ for a vector in full
     * storage format.
     */
    int64_t num_nonzeros;

    /**
     * ‘data’ contains the data lines of the vector.
     */
    union {
        float * real_single;
        double * real_double;
        float (* complex_single)[2];
        double (* complex_double)[2];
        int32_t * integer_single;
        int64_t * integer_double;
        void * pattern;
    } data;

    /**
     * ‘idx’ is an array of length ‘num_nonzeros’, containing the
     * offset of each nonzero vector entry. Note that offsets are
     * 0-based, unlike the Matrix Market format, where indices are
     * 1-based.
     *
     * Note that ‘idx’ is set to ‘NULL’ for vectors in full storage
     * format. In this case, ‘size’ and ‘num_nonzeros’ must be equal,
     * and elements of the vector are implicitly numbered from ‘0’ up
     * to ‘size-1’.
     */
    int64_t * idx;
};

/*
 * vector properties
 */

/**
 * ‘mtxbasevector_field()’ gets the field of a vector.
 */
LIBMTX_API enum mtxfield mtxbasevector_field(const struct mtxbasevector * x);

/**
 * ‘mtxbasevector_precision()’ gets the precision of a vector.
 */
LIBMTX_API enum mtxprecision mtxbasevector_precision(const struct mtxbasevector * x);

/**
 * ‘mtxbasevector_size()’ gets the size of a vector.
 */
LIBMTX_API int64_t mtxbasevector_size(const struct mtxbasevector * x);

/**
 * ‘mtxbasevector_num_nonzeros()’ gets the number of explicitly
 * stored vector entries.
 */
LIBMTX_API int64_t mtxbasevector_num_nonzeros(const struct mtxbasevector * x);

/**
 * ‘mtxbasevector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
LIBMTX_API int64_t * mtxbasevector_idx(const struct mtxbasevector * x);

/*
 * memory management
 */

/**
 * ‘mtxbasevector_free()’ frees storage allocated for a vector.
 */
void LIBMTX_API mtxbasevector_free(
    struct mtxbasevector * x);

/**
 * ‘mtxbasevector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int LIBMTX_API mtxbasevector_alloc_copy(
    struct mtxbasevector * dst,
    const struct mtxbasevector * src);

/**
 * ‘mtxbasevector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int LIBMTX_API mtxbasevector_init_copy(
    struct mtxbasevector * dst,
    const struct mtxbasevector * src);

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxbasevector_alloc()’ allocates a vector.
 */
int LIBMTX_API mtxbasevector_alloc(
    struct mtxbasevector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxbasevector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_real_single(
    struct mtxbasevector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxbasevector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_real_double(
    struct mtxbasevector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxbasevector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxbasevector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxbasevector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxbasevector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxbasevector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int LIBMTX_API mtxbasevector_init_pattern(
    struct mtxbasevector * x,
    int64_t size);

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxbasevector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_strided_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const float * data);

/**
 * ‘mtxbasevector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_strided_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const double * data);

/**
 * ‘mtxbasevector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_strided_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2]);

/**
 * ‘mtxbasevector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_strided_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2]);

/**
 * ‘mtxbasevector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_strided_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data);

/**
 * ‘mtxbasevector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_strided_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data);

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxbasevector_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int LIBMTX_API mtxbasevector_alloc_packed(
    struct mtxbasevector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/**
 * ‘mtxbasevector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data);

/**
 * ‘mtxbasevector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data);

/**
 * ‘mtxbasevector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2]);

/**
 * ‘mtxbasevector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2]);

/**
 * ‘mtxbasevector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data);

/**
 * ‘mtxbasevector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data);

/**
 * ‘mtxbasevector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int LIBMTX_API mtxbasevector_init_packed_pattern(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxbasevector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int LIBMTX_API mtxbasevector_alloc_packed_strided(
    struct mtxbasevector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/**
 * ‘mtxbasevector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data);

/**
 * ‘mtxbasevector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data);

/**
 * ‘mtxbasevector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2]);

/**
 * ‘mtxbasevector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2]);

/**
 * ‘mtxbasevector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data);

/**
 * ‘mtxbasevector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data);

/**
 * ‘mtxbasevector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int LIBMTX_API mtxbasevector_init_packed_strided_pattern(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/*
 * accessing values
 */

/**
 * ‘mtxbasevector_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int LIBMTX_API mtxbasevector_get_real_single(
    const struct mtxbasevector * x,
    int64_t size,
    int stride,
    float * a);

/**
 * ‘mtxbasevector_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int LIBMTX_API mtxbasevector_get_real_double(
    const struct mtxbasevector * x,
    int64_t size,
    int stride,
    double * a);

/**
 * ‘mtxbasevector_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int LIBMTX_API mtxbasevector_get_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    float (* a)[2]);

/**
 * ‘mtxbasevector_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int LIBMTX_API mtxbasevector_get_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    double (* a)[2]);

/**
 * ‘mtxbasevector_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int LIBMTX_API mtxbasevector_get_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    int32_t * a);

/**
 * ‘mtxbasevector_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int LIBMTX_API mtxbasevector_get_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    int64_t * a);

/*
 * modifying values
 */

/**
 * ‘mtxbasevector_setzero()’ sets every value of a vector to zero.
 */
int LIBMTX_API mtxbasevector_setzero(
    struct mtxbasevector * x);

/**
 * ‘mtxbasevector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int LIBMTX_API mtxbasevector_set_constant_real_single(
    struct mtxbasevector * x,
    float a);

/**
 * ‘mtxbasevector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int LIBMTX_API mtxbasevector_set_constant_real_double(
    struct mtxbasevector * x,
    double a);

/**
 * ‘mtxbasevector_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int LIBMTX_API mtxbasevector_set_constant_complex_single(
    struct mtxbasevector * x,
    float a[2]);

/**
 * ‘mtxbasevector_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int LIBMTX_API mtxbasevector_set_constant_complex_double(
    struct mtxbasevector * x,
    double a[2]);

/**
 * ‘mtxbasevector_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int LIBMTX_API mtxbasevector_set_constant_integer_single(
    struct mtxbasevector * x,
    int32_t a);

/**
 * ‘mtxbasevector_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int LIBMTX_API mtxbasevector_set_constant_integer_double(
    struct mtxbasevector * x,
    int64_t a);

/**
 * ‘mtxbasevector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int LIBMTX_API mtxbasevector_set_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxbasevector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int LIBMTX_API mtxbasevector_set_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxbasevector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int LIBMTX_API mtxbasevector_set_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxbasevector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int LIBMTX_API mtxbasevector_set_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxbasevector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int LIBMTX_API mtxbasevector_set_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxbasevector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int LIBMTX_API mtxbasevector_set_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxbasevector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int LIBMTX_API mtxbasevector_from_mtxfile(
    struct mtxbasevector * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxbasevector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int LIBMTX_API mtxbasevector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxbasevector * x,
    int64_t /* num_rows */,
    const int64_t * /* idx */,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxbasevector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxbasevector’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxbasevector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int LIBMTX_API mtxbasevector_split(
    int num_parts,
    struct mtxbasevector ** dsts,
    const struct mtxbasevector * src,
    int64_t size,
    int * parts,
    int64_t * invperm);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxbasevector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_swap(
    struct mtxbasevector * x,
    struct mtxbasevector * y);

/**
 * ‘mtxbasevector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_copy(
    struct mtxbasevector * y,
    const struct mtxbasevector * x);

/**
 * ‘mtxbasevector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int LIBMTX_API mtxbasevector_sscal(
    float a,
    struct mtxbasevector * x,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int LIBMTX_API mtxbasevector_dscal(
    double a,
    struct mtxbasevector * x,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int LIBMTX_API mtxbasevector_cscal(
    float a[2],
    struct mtxbasevector * x,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int LIBMTX_API mtxbasevector_zscal(
    double a[2],
    struct mtxbasevector * x,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_saxpy(
    float a,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_daxpy(
    double a,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_caxpy(
    float a[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_zaxpy(
    double a[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_saypx(
    float a,
    struct mtxbasevector * y,
    const struct mtxbasevector * x,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_daypx(
    double a,
    struct mtxbasevector * y,
    const struct mtxbasevector * x,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_sdot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_ddot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_cdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_cdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_zdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_zdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_cdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int LIBMTX_API mtxbasevector_zdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_snrm2()’ computes the Euclidean norm of a vector
 * in single precision floating point.
 */
int LIBMTX_API mtxbasevector_snrm2(
    const struct mtxbasevector * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_dnrm2()’ computes the Euclidean norm of a vector
 * in double precision floating point.
 */
int LIBMTX_API mtxbasevector_dnrm2(
    const struct mtxbasevector * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int LIBMTX_API mtxbasevector_sasum(
    const struct mtxbasevector * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int LIBMTX_API mtxbasevector_dasum(
    const struct mtxbasevector * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int LIBMTX_API mtxbasevector_iamax(
    const struct mtxbasevector * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxbasevector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int LIBMTX_API mtxbasevector_ussdot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int LIBMTX_API mtxbasevector_usddot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int LIBMTX_API mtxbasevector_uscdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int LIBMTX_API mtxbasevector_uszdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int LIBMTX_API mtxbasevector_uscdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int LIBMTX_API mtxbasevector_uszdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxbasevector_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int LIBMTX_API mtxbasevector_ussaxpy(
    float alpha,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int LIBMTX_API mtxbasevector_usdaxpy(
    double alpha,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int LIBMTX_API mtxbasevector_uscaxpy(
    float alpha[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int LIBMTX_API mtxbasevector_uszaxpy(
    double alpha[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops);

/**
 * ‘mtxbasevector_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int LIBMTX_API mtxbasevector_usga(
    struct mtxbasevector * x,
    const struct mtxbasevector * y);

/**
 * ‘mtxbasevector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int LIBMTX_API mtxbasevector_usgz(
    struct mtxbasevector * x,
    struct mtxbasevector * y);

/**
 * ‘mtxbasevector_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int LIBMTX_API mtxbasevector_ussc(
    struct mtxbasevector * y,
    const struct mtxbasevector * x);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxbasevector_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int LIBMTX_API mtxbasevector_usscga(
    struct mtxbasevector * z,
    const struct mtxbasevector * x);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxbasevector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxbasevector_recv()’.
 */
int LIBMTX_API mtxbasevector_send(
    const struct mtxbasevector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxbasevector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxbasevector_send()’.
 */
int LIBMTX_API mtxbasevector_recv(
    struct mtxbasevector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode);

/**
 * ‘mtxbasevector_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxbasevector_send()’.
 */
int LIBMTX_API mtxbasevector_irecv(
    struct mtxbasevector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode);
#endif

#endif
