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
 * Data structures and routines for shared-memory parallel, dense
 * vectors using OpenMP.
 */

#ifndef LIBMTX_LINALG_OMP_VECTOR_H
#define LIBMTX_LINALG_OMP_VECTOR_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_OPENMP
#include <libmtx/mtxfile/header.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/precision.h>

#include <omp.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxvector;

/**
 * ‘mtxompvector’ represents a dense vector shared among one or more
 * threads using OpenMP.
 */
struct mtxompvector
{
    /**
     * ‘num_threads’ is the maximum number of OpenMP threads to use
     * when carrying out operations on the underlying vector.
     */
    int num_threads;

    /**
     * ‘offsets’ is an optional array that may be used to specify a
     * variable-sized block distribution of vector elements.
     *
     * Whenever ‘offsets’ is not ‘NULL’, then it is an array of length
     * ‘num_threads+1’ containing non-decreasing offsets in the range
     * ‘0’ up to ‘N’, where ‘N’ is the number of vector elements. For
     * each thread, the array stores the offset to the first vector
     * element assigned to the thread. The final value in the array
     * must be equal to the number of vector elements ‘N’.
     *
     * If ‘offsets’ is ‘NULL’, then parallel loops instead use a
     * schedule determined by ‘sched’ and ‘chunk_size’.
     */
    int64_t * offsets;

    /**
     * ‘sched’ is the schedule to use for parallel loops.
     *
     * This is used only if ‘offsets’ is ‘NULL’.
     */
    omp_sched_t sched;

    /**
     * ‘chunk_size’ is the chunk size to use for parallel loops.
     *
     * This is used only if ‘offsets’ is ‘NULL’. If ‘chunk_size’ is
     * set to ‘0’, a default chunk size is used for the given
     * schedule.
     */
    int chunk_size;

    /**
     * ‘base’ is the underlying dense vector.
     */
    struct mtxbasevector base;
};

/*
 * vector properties
 */

/**
 * ‘mtxompvector_field()’ gets the field of a vector.
 */
enum mtxfield mtxompvector_field(const struct mtxompvector * x);

/**
 * ‘mtxompvector_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxompvector_precision(const struct mtxompvector * x);

/**
 * ‘mtxompvector_size()’ gets the size of a vector.
 */
int64_t mtxompvector_size(const struct mtxompvector * x);

/**
 * ‘mtxompvector_num_nonzeros()’ gets the number of explicitly stored
 * vector entries.
 */
int64_t mtxompvector_num_nonzeros(const struct mtxompvector * x);

/**
 * ‘mtxompvector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxompvector_idx(const struct mtxompvector * x);

/*
 * memory management
 */

/**
 * ‘mtxompvector_free()’ frees storage allocated for a vector.
 */
void mtxompvector_free(
    struct mtxompvector * x);

/**
 * ‘mtxompvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxompvector_alloc_copy(
    struct mtxompvector * dst,
    const struct mtxompvector * src);

/**
 * ‘mtxompvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxompvector_init_copy(
    struct mtxompvector * dst,
    const struct mtxompvector * src);

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxompvector_alloc()’ allocates a vector.
 *
 * A default static schedule is used for parallel loops.
 */
int mtxompvector_alloc(
    struct mtxompvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxompvector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxompvector_init_real_single(
    struct mtxompvector * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxompvector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxompvector_init_real_double(
    struct mtxompvector * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxompvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxompvector_init_complex_single(
    struct mtxompvector * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxompvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxompvector_init_complex_double(
    struct mtxompvector * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxompvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxompvector_init_integer_single(
    struct mtxompvector * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxompvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxompvector_init_integer_double(
    struct mtxompvector * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxompvector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxompvector_init_pattern(
    struct mtxompvector * x,
    int64_t size);

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxompvector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxompvector_init_strided_real_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const float * data);

/**
 * ‘mtxompvector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxompvector_init_strided_real_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const double * data);

/**
 * ‘mtxompvector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxompvector_init_strided_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2]);

/**
 * ‘mtxompvector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxompvector_init_strided_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2]);

/**
 * ‘mtxompvector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxompvector_init_strided_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data);

/**
 * ‘mtxompvector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxompvector_init_strided_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data);

/*
 * allocation and initialisation with custom schedule
 */

/**
 * ‘mtxompvector_alloc_custom()’ allocates a vector with a
 * user-defined schedule for parallel loops.
 *
 * If ‘offsets’ is ‘NULL’, then it is ignored. In this case, parallel
 * loops employ a user-defined schedule and chunk size, as specified
 * by ‘sched’ and ‘chunk_size’.
 *
 * Otherwise, a variable-sized block distribution of vector elements
 * is used. In this case, ‘offsets’ must point to an array of length
 * ‘num_threads+1’, containing the offsets to the first vector element
 * assigned to each thread. Moreover, ‘offsets[num_threads]’ must be
 * equal to the total number of vector elements.
 */
int mtxompvector_alloc_custom(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxompvector_init_custom_real_single()’ allocates and initialises
 * a vector with real, single precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_real_single(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float * data);

/**
 * ‘mtxompvector_init_custom_real_double()’ allocates and initialises
 * a vector with real, double precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_real_double(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double * data);

/**
 * ‘mtxompvector_init_custom_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_complex_single(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float (* data)[2]);

/**
 * ‘mtxompvector_init_custom_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_complex_double(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double (* data)[2]);

/**
 * ‘mtxompvector_init_custom_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_integer_single(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int32_t * data);

/**
 * ‘mtxompvector_init_custom_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_integer_double(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int64_t * data);

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxompvector_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxompvector_alloc_packed(
    struct mtxompvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/**
 * ‘mtxompvector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxompvector_init_packed_real_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data);

/**
 * ‘mtxompvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxompvector_init_packed_real_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data);

/**
 * ‘mtxompvector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxompvector_init_packed_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2]);

/**
 * ‘mtxompvector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxompvector_init_packed_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2]);

/**
 * ‘mtxompvector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxompvector_init_packed_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data);

/**
 * ‘mtxompvector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxompvector_init_packed_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data);

/**
 * ‘mtxompvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxompvector_init_packed_pattern(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxompvector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxompvector_alloc_packed_strided(
    struct mtxompvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/**
 * ‘mtxompvector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxompvector_init_packed_strided_real_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data);

/**
 * ‘mtxompvector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxompvector_init_packed_strided_real_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data);

/**
 * ‘mtxompvector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxompvector_init_packed_strided_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2]);

/**
 * ‘mtxompvector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxompvector_init_packed_strided_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2]);

/**
 * ‘mtxompvector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxompvector_init_packed_strided_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data);

/**
 * ‘mtxompvector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxompvector_init_packed_strided_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data);

/**
 * ‘mtxompvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxompvector_init_packed_strided_pattern(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/*
 * accessing values
 */

/**
 * ‘mtxompvector_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_real_single(
    const struct mtxompvector * x,
    int64_t size,
    int stride,
    float * a);

/**
 * ‘mtxompvector_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_real_double(
    const struct mtxompvector * x,
    int64_t size,
    int stride,
    double * a);

/**
 * ‘mtxompvector_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    float (* a)[2]);

/**
 * ‘mtxompvector_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    double (* a)[2]);

/**
 * ‘mtxompvector_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    int32_t * a);

/**
 * ‘mtxompvector_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    int64_t * a);

/*
 * Modifying values
 */

/**
 * ‘mtxompvector_setzero()’ sets every value of a vector to zero.
 */
int mtxompvector_setzero(
    struct mtxompvector * x);

/**
 * ‘mtxompvector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxompvector_set_constant_real_single(
    struct mtxompvector * x,
    float a);

/**
 * ‘mtxompvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxompvector_set_constant_real_double(
    struct mtxompvector * x,
    double a);

/**
 * ‘mtxompvector_set_constant_complex_single()’ sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtxompvector_set_constant_complex_single(
    struct mtxompvector * x,
    float a[2]);

/**
 * ‘mtxompvector_set_constant_complex_double()’ sets every value of a
 * vector equal to a constant, double precision floating point complex
 * number.
 */
int mtxompvector_set_constant_complex_double(
    struct mtxompvector * x,
    double a[2]);

/**
 * ‘mtxompvector_set_constant_integer_single()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxompvector_set_constant_integer_single(
    struct mtxompvector * x,
    int32_t a);

/**
 * ‘mtxompvector_set_constant_integer_double()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxompvector_set_constant_integer_double(
    struct mtxompvector * x,
    int64_t a);

/**
 * ‘mtxompvector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxompvector_set_real_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxompvector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxompvector_set_real_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxompvector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxompvector_set_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxompvector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxompvector_set_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxompvector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxompvector_set_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxompvector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxompvector_set_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxompvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxompvector_from_mtxfile(
    struct mtxompvector * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxompvector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxompvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxompvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxompvector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxompvector’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxompvector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxompvector_split(
    int num_parts,
    struct mtxompvector ** dsts,
    const struct mtxompvector * src,
    int64_t size,
    int * parts,
    int64_t * invperm);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxompvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_swap(
    struct mtxompvector * x,
    struct mtxompvector * y);

/**
 * ‘mtxompvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_copy(
    struct mtxompvector * y,
    const struct mtxompvector * x);

/**
 * ‘mtxompvector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxompvector_sscal(
    float a,
    struct mtxompvector * x,
    int64_t * num_flops);

/**
 * ‘mtxompvector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxompvector_dscal(
    double a,
    struct mtxompvector * x,
    int64_t * num_flops);

/**
 * ‘mtxompvector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxompvector_cscal(
    float a[2],
    struct mtxompvector * x,
    int64_t * num_flops);

/**
 * ‘mtxompvector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxompvector_zscal(
    double a[2],
    struct mtxompvector * x,
    int64_t * num_flops);

/**
 * ‘mtxompvector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_saxpy(
    float a,
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_daxpy(
    double a,
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);


/**
 * ‘mtxompvector_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_caxpy(
    float a[2],
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_zaxpy(
    double a[2],
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_saypx(
    float a,
    struct mtxompvector * y,
    const struct mtxompvector * x,
    int64_t * num_flops);

/**
 * ‘mtxompvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_daypx(
    double a,
    struct mtxompvector * y,
    const struct mtxompvector * x,
    int64_t * num_flops);

/**
 * ‘mtxompvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_sdot(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxompvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_ddot(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxompvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_cdotu(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_zdotu(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_cdotc(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_zdotc(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxompvector_snrm2(
    const struct mtxompvector * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxompvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxompvector_dnrm2(
    const struct mtxompvector * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxompvector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxompvector_sasum(
    const struct mtxompvector * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxompvector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxompvector_dasum(
    const struct mtxompvector * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxompvector_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the vector is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxompvector_iamax(
    const struct mtxompvector * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxompvector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_ussdot(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxompvector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_usddot(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxompvector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uscdotu(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uszdotu(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uscdotc(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uszdotc(
    const struct mtxompvector * x,
    const struct mtxompvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxompvector_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_ussaxpy(
    float alpha,
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_usdaxpy(
    double alpha,
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_uscaxpy(
    float alpha[2],
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_uszaxpy(
    double alpha[2],
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops);

/**
 * ‘mtxompvector_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxompvector_usga(
    struct mtxompvector * x,
    const struct mtxompvector * y);

/**
 * ‘mtxompvector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxompvector_usgz(
    struct mtxompvector * x,
    struct mtxompvector * y);

/**
 * ‘mtxompvector_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxompvector_ussc(
    struct mtxompvector * y,
    const struct mtxompvector * x);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxompvector_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxompvector_usscga(
    struct mtxompvector * z,
    const struct mtxompvector * x);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxompvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxompvector_recv()’.
 */
int mtxompvector_send(
    const struct mtxompvector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxompvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxompvector_send()’.
 */
int mtxompvector_recv(
    struct mtxompvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode);

/**
 * ‘mtxompvector_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxompvector_send()’.
 */
int mtxompvector_irecv(
    struct mtxompvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode);
#endif
#endif

#endif
