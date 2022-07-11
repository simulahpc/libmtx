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
 * Data structures and routines for shared-memory parallel, dense
 * vectors using OpenMP.
 */

#ifndef LIBMTX_VECTOR_OMP_H
#define LIBMTX_VECTOR_OMP_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_OPENMP
#include <libmtx/mtxfile/header.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/field.h>
#include <libmtx/vector/precision.h>

#include <omp.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxvector;

/**
 * ‘mtxvector_omp’ represents a dense vector shared among one or more
 * threads using OpenMP.
 */
struct mtxvector_omp
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
    struct mtxvector_base base;
};

/*
 * vector properties
 */

/**
 * ‘mtxvector_omp_field()’ gets the field of a vector.
 */
enum mtxfield mtxvector_omp_field(const struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxvector_omp_precision(const struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_size()’ gets the size of a vector.
 */
int64_t mtxvector_omp_size(const struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_num_nonzeros()’ gets the number of explicitly stored
 * vector entries.
 */
int64_t mtxvector_omp_num_nonzeros(const struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxvector_omp_idx(const struct mtxvector_omp * x);

/*
 * memory management
 */

/**
 * ‘mtxvector_omp_free()’ frees storage allocated for a vector.
 */
void mtxvector_omp_free(
    struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_omp_alloc_copy(
    struct mtxvector_omp * dst,
    const struct mtxvector_omp * src);

/**
 * ‘mtxvector_omp_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_omp_init_copy(
    struct mtxvector_omp * dst,
    const struct mtxvector_omp * src);

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxvector_omp_alloc()’ allocates a vector.
 *
 * A default static schedule is used for parallel loops.
 */
int mtxvector_omp_alloc(
    struct mtxvector_omp * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_omp_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_omp_init_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxvector_omp_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_omp_init_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxvector_omp_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxvector_omp_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxvector_omp_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxvector_omp_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxvector_omp_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_omp_init_pattern(
    struct mtxvector_omp * x,
    int64_t size);

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxvector_omp_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_omp_init_strided_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float * data);

/**
 * ‘mtxvector_omp_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_omp_init_strided_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const double * data);

/**
 * ‘mtxvector_omp_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_strided_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2]);

/**
 * ‘mtxvector_omp_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_strided_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2]);

/**
 * ‘mtxvector_omp_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_strided_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const int32_t * data);

/**
 * ‘mtxvector_omp_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_strided_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const int64_t * data);

/*
 * allocation and initialisation with custom schedule
 */

/**
 * ‘mtxvector_omp_alloc_custom()’ allocates a vector with a
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
int mtxvector_omp_alloc_custom(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_omp_init_custom_real_single()’ allocates and initialises
 * a vector with real, single precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_real_single(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float * data);

/**
 * ‘mtxvector_omp_init_custom_real_double()’ allocates and initialises
 * a vector with real, double precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_real_double(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double * data);

/**
 * ‘mtxvector_omp_init_custom_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_complex_single(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float (* data)[2]);

/**
 * ‘mtxvector_omp_init_custom_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_complex_double(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double (* data)[2]);

/**
 * ‘mtxvector_omp_init_custom_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_integer_single(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int32_t * data);

/**
 * ‘mtxvector_omp_init_custom_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_integer_double(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxvector_omp_alloc_packed(
    struct mtxvector_omp * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/**
 * ‘mtxvector_omp_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_omp_init_packed_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data);

/**
 * ‘mtxvector_omp_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_omp_init_packed_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data);

/**
 * ‘mtxvector_omp_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_packed_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2]);

/**
 * ‘mtxvector_omp_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_packed_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2]);

/**
 * ‘mtxvector_omp_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_packed_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data);

/**
 * ‘mtxvector_omp_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_packed_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data);

/**
 * ‘mtxvector_omp_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxvector_omp_init_packed_pattern(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxvector_omp_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxvector_omp_alloc_packed_strided(
    struct mtxvector_omp * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/**
 * ‘mtxvector_omp_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_omp_init_packed_strided_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data);

/**
 * ‘mtxvector_omp_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_omp_init_packed_strided_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data);

/**
 * ‘mtxvector_omp_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_packed_strided_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2]);

/**
 * ‘mtxvector_omp_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_packed_strided_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2]);

/**
 * ‘mtxvector_omp_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_packed_strided_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data);

/**
 * ‘mtxvector_omp_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_packed_strided_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data);

/**
 * ‘mtxvector_omp_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxvector_omp_init_packed_strided_pattern(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx);

/*
 * accessing values
 */

/**
 * ‘mtxvector_omp_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_real_single(
    const struct mtxvector_omp * x,
    int64_t size,
    int stride,
    float * a);

/**
 * ‘mtxvector_omp_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_real_double(
    const struct mtxvector_omp * x,
    int64_t size,
    int stride,
    double * a);

/**
 * ‘mtxvector_omp_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    float (* a)[2]);

/**
 * ‘mtxvector_omp_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    double (* a)[2]);

/**
 * ‘mtxvector_omp_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    int32_t * a);

/**
 * ‘mtxvector_omp_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    int64_t * a);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_omp_setzero()’ sets every value of a vector to zero.
 */
int mtxvector_omp_setzero(
    struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_omp_set_constant_real_single(
    struct mtxvector_omp * x,
    float a);

/**
 * ‘mtxvector_omp_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_omp_set_constant_real_double(
    struct mtxvector_omp * x,
    double a);

/**
 * ‘mtxvector_omp_set_constant_complex_single()’ sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtxvector_omp_set_constant_complex_single(
    struct mtxvector_omp * x,
    float a[2]);

/**
 * ‘mtxvector_omp_set_constant_complex_double()’ sets every value of a
 * vector equal to a constant, double precision floating point complex
 * number.
 */
int mtxvector_omp_set_constant_complex_double(
    struct mtxvector_omp * x,
    double a[2]);

/**
 * ‘mtxvector_omp_set_constant_integer_single()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxvector_omp_set_constant_integer_single(
    struct mtxvector_omp * x,
    int32_t a);

/**
 * ‘mtxvector_omp_set_constant_integer_double()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxvector_omp_set_constant_integer_double(
    struct mtxvector_omp * x,
    int64_t a);

/**
 * ‘mtxvector_omp_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxvector_omp_set_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxvector_omp_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxvector_omp_set_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxvector_omp_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxvector_omp_set_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxvector_omp_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxvector_omp_set_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxvector_omp_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_omp_set_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxvector_omp_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_omp_set_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int stride,
    const int64_t * a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_omp_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_omp_from_mtxfile(
    struct mtxvector_omp * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxvector_omp_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_omp_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_omp * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_omp_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxvector_omp’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxvector_omp_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxvector_omp_split(
    int num_parts,
    struct mtxvector_omp ** dsts,
    const struct mtxvector_omp * src,
    int64_t size,
    int * parts,
    int64_t * invperm);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_omp_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_swap(
    struct mtxvector_omp * x,
    struct mtxvector_omp * y);

/**
 * ‘mtxvector_omp_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_copy(
    struct mtxvector_omp * y,
    const struct mtxvector_omp * x);

/**
 * ‘mtxvector_omp_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_omp_sscal(
    float a,
    struct mtxvector_omp * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_omp_dscal(
    double a,
    struct mtxvector_omp * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_omp_cscal(
    float a[2],
    struct mtxvector_omp * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_omp_zscal(
    double a[2],
    struct mtxvector_omp * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_saxpy(
    float a,
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_daxpy(
    double a,
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);


/**
 * ‘mtxvector_omp_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_caxpy(
    float a[2],
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_zaxpy(
    double a[2],
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_saypx(
    float a,
    struct mtxvector_omp * y,
    const struct mtxvector_omp * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_daypx(
    double a,
    struct mtxvector_omp * y,
    const struct mtxvector_omp * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_sdot(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_ddot(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_cdotu(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_zdotu(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_cdotc(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_zdotc(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_omp_snrm2(
    const struct mtxvector_omp * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_omp_dnrm2(
    const struct mtxvector_omp * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_omp_sasum(
    const struct mtxvector_omp * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_omp_dasum(
    const struct mtxvector_omp * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the vector is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxvector_omp_iamax(
    const struct mtxvector_omp * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_omp_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_ussdot(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_usddot(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uscdotu(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uszdotu(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uscdotc(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uszdotc(
    const struct mtxvector_omp * x,
    const struct mtxvector_omp * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_ussaxpy(
    float alpha,
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_usdaxpy(
    double alpha,
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_uscaxpy(
    float alpha[2],
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_uszaxpy(
    double alpha[2],
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxvector_omp_usga(
    struct mtxvector_omp * x,
    const struct mtxvector_omp * y);

/**
 * ‘mtxvector_omp_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxvector_omp_usgz(
    struct mtxvector_omp * x,
    struct mtxvector_omp * y);

/**
 * ‘mtxvector_omp_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_omp_ussc(
    struct mtxvector_omp * y,
    const struct mtxvector_omp * x);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxvector_omp_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxvector_omp_usscga(
    struct mtxvector_omp * z,
    const struct mtxvector_omp * x);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_omp_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_omp_recv()’.
 */
int mtxvector_omp_send(
    const struct mtxvector_omp * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxvector_omp_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_omp_send()’.
 */
int mtxvector_omp_recv(
    struct mtxvector_omp * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode);

/**
 * ‘mtxvector_omp_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_omp_send()’.
 */
int mtxvector_omp_irecv(
    struct mtxvector_omp * x,
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
