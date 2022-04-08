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
 * Last modified: 2022-04-08
 *
 * Data structures and routines for basic dense vectors.
 */

#ifndef LIBMTX_VECTOR_BASE_H
#define LIBMTX_VECTOR_BASE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/vector/vector_array.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxpartition;
struct mtxvector;

/**
 * ‘mtxvector_base’ represents a dense vector stored as a contiguous
 * array of elements.
 */
struct mtxvector_base
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
};

/*
 * Memory management
 */

/**
 * ‘mtxvector_base_free()’ frees storage allocated for a vector.
 */
void mtxvector_base_free(
    struct mtxvector_base * x);

/**
 * ‘mtxvector_base_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_base_alloc_copy(
    struct mtxvector_base * dst,
    const struct mtxvector_base * src);

/**
 * ‘mtxvector_base_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_base_init_copy(
    struct mtxvector_base * dst,
    const struct mtxvector_base * src);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_base_alloc()’ allocates a vector.
 */
int mtxvector_base_alloc(
    struct mtxvector_base * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_base_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_base_init_real_single(
    struct mtxvector_base * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxvector_base_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_base_init_real_double(
    struct mtxvector_base * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxvector_base_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_base_init_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxvector_base_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_base_init_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxvector_base_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_base_init_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxvector_base_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_base_init_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxvector_base_init_pattern()’ allocates and initialises a binary
 * pattern vector, where every entry has a value of one.
 */
int mtxvector_base_init_pattern(
    struct mtxvector_base * x,
    int64_t size);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_base_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_base_set_constant_real_single(
    struct mtxvector_base * x,
    float a);

/**
 * ‘mtxvector_base_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_base_set_constant_real_double(
    struct mtxvector_base * x,
    double a);

/**
 * ‘mtxvector_base_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_base_set_constant_complex_single(
    struct mtxvector_base * x,
    float a[2]);

/**
 * ‘mtxvector_base_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_base_set_constant_complex_double(
    struct mtxvector_base * x,
    double a[2]);

/**
 * ‘mtxvector_base_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_base_set_constant_integer_single(
    struct mtxvector_base * x,
    int32_t a);

/**
 * ‘mtxvector_base_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_base_set_constant_integer_double(
    struct mtxvector_base * x,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_base_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_base_from_mtxfile(
    struct mtxvector_base * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxvector_base_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_base_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_base * x,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_base_partition()’ partitions a vector into blocks
 * according to the given partitioning.
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
int mtxvector_base_partition(
    struct mtxvector * dsts,
    const struct mtxvector_base * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_base_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_base_join(
    struct mtxvector_base * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_base_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_swap(
    struct mtxvector_base * x,
    struct mtxvector_base * y);

/**
 * ‘mtxvector_base_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_copy(
    struct mtxvector_base * y,
    const struct mtxvector_base * x);

/**
 * ‘mtxvector_base_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_base_sscal(
    float a,
    struct mtxvector_base * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_base_dscal(
    double a,
    struct mtxvector_base * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_base_cscal(
    float a[2],
    struct mtxvector_base * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_base_zscal(
    double a[2],
    struct mtxvector_base * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_saxpy(
    float a,
    const struct mtxvector_base * x,
    struct mtxvector_base * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_daxpy(
    double a,
    const struct mtxvector_base * x,
    struct mtxvector_base * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_saypx(
    float a,
    struct mtxvector_base * y,
    const struct mtxvector_base * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_daypx(
    double a,
    struct mtxvector_base * y,
    const struct mtxvector_base * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_sdot()’ cbaseutes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_sdot(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_ddot()’ cbaseutes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_ddot(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_cdotu()’ cbaseutes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_cdotu(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_zdotu()’ cbaseutes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_zdotu(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_cdotc()’ cbaseutes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_cdotc(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_zdotc()’ cbaseutes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_zdotc(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_snrm2()’ cbaseutes the Euclidean norm of a vector
 * in single precision floating point.
 */
int mtxvector_base_snrm2(
    const struct mtxvector_base * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_dnrm2()’ cbaseutes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxvector_base_dnrm2(
    const struct mtxvector_base * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_sasum()’ cbaseutes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_sasum(
    const struct mtxvector_base * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_dasum()’ cbaseutes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_dasum(
    const struct mtxvector_base * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_base_iamax(
    const struct mtxvector_base * x,
    int * iamax);

#endif