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
 * Last modified: 2022-04-09
 *
 * Data structures and routines for sparse vectors in packed storage
 * format.
 */

#ifndef LIBMTX_VECTOR_PACKED_H
#define LIBMTX_VECTOR_PACKED_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/vector/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxpartition;
struct mtxvector;

/**
 * ‘mtxvector_packed’ represents a sparse vector in a packed storage
 * format.
 *
 * The vector is thus represented by a contiguous array of elements
 * together with an array of integers designating the offset of each
 * element.
 */
struct mtxvector_packed
{
    /**
     * ‘size’ is the number of vector elements.
     */
    int64_t size;

    /**
     * ‘num_nonzeros’ is the number of explicitly stored vector
     * entries for a sparse vector in packed storage format.
     */
    int64_t num_nonzeros;

    /**
     * ‘idx’ is an array of length ‘num_nonzeros’, containing the
     * offset of each nonzero vector entry. Note that offsets are
     * 0-based, unlike the Matrix Market format, where indices are
     * 1-based.
     */
    int64_t * idx;

    /**
     * ‘x’ is the underlying storage of the nonzero vector elements.
     */
    struct mtxvector x;
};

/*
 * Memory management
 */

/**
 * ‘mtxvector_packed_free()’ frees storage allocated for a vector.
 */
void mtxvector_packed_free(
    struct mtxvector_packed * x);

/**
 * ‘mtxvector_packed_alloc_copy()’ allocates a copy of a vector
 * without initialising the values.
 */
int mtxvector_packed_alloc_copy(
    struct mtxvector_packed * dst,
    const struct mtxvector_packed * src);

/**
 * ‘mtxvector_packed_init_copy()’ allocates a copy of a vector and
 * also copies the values.
 */
int mtxvector_packed_init_copy(
    struct mtxvector_packed * dst,
    const struct mtxvector_packed * src);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_packed_alloc()’ allocates a sparse vector in packed
 * storage format, where nonzero coefficients are stored in an
 * underlying dense vector of the given type.
 */
int mtxvector_packed_alloc(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros);

/**
 * ‘mtxvector_packed_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_packed_init_real_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data);

/**
 * ‘mtxvector_packed_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_packed_init_real_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data);

/**
 * ‘mtxvector_packed_init_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxvector_packed_init_complex_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2]);

/**
 * ‘mtxvector_packed_init_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxvector_packed_init_complex_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2]);

/**
 * ‘mtxvector_packed_init_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxvector_packed_init_integer_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data);

/**
 * ‘mtxvector_packed_init_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxvector_packed_init_integer_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data);

/**
 * ‘mtxvector_packed_init_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxvector_packed_init_pattern(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_packed_set_constant_real_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point number.
 */
int mtxvector_packed_set_constant_real_single(
    struct mtxvector_packed * x,
    float a);

/**
 * ‘mtxvector_packed_set_constant_real_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point number.
 */
int mtxvector_packed_set_constant_real_double(
    struct mtxvector_packed * x,
    double a);

/**
 * ‘mtxvector_packed_set_constant_complex_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxvector_packed_set_constant_complex_single(
    struct mtxvector_packed * x,
    float a[2]);

/**
 * ‘mtxvector_packed_set_constant_complex_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxvector_packed_set_constant_complex_double(
    struct mtxvector_packed * x,
    double a[2]);

/**
 * ‘mtxvector_packed_set_constant_integer_single()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxvector_packed_set_constant_integer_single(
    struct mtxvector_packed * x,
    int32_t a);

/**
 * ‘mtxvector_packed_set_constant_integer_double()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxvector_packed_set_constant_integer_double(
    struct mtxvector_packed * x,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_packed_from_mtxfile()’ converts from a vector in Matrix
 * Market format.
 */
int mtxvector_packed_from_mtxfile(
    struct mtxvector_packed * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxvector_packed_to_mtxfile()’ converts to a vector in Matrix
 * Market format.
 */
int mtxvector_packed_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_packed * x,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_packed_partition()’ partitions a vector into blocks
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
int mtxvector_packed_partition(
    struct mtxvector * dsts,
    const struct mtxvector_packed * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_packed_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_packed_join(
    struct mtxvector_packed * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_packed_swap()’ swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_swap(
    struct mtxvector_packed * x,
    struct mtxvector_packed * y);

/**
 * ‘mtxvector_packed_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_copy(
    struct mtxvector_packed * y,
    const struct mtxvector_packed * x);

/**
 * ‘mtxvector_packed_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_packed_sscal(
    float a,
    struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_packed_dscal(
    double a,
    struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_packed_cscal(
    float a[2],
    struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_packed_zscal(
    double a[2],
    struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_saxpy()’ adds a vector to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_saxpy(
    float a,
    const struct mtxvector_packed * x,
    struct mtxvector_packed * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_daxpy()’ adds a vector to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_daxpy(
    double a,
    const struct mtxvector_packed * x,
    struct mtxvector_packed * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_saypx()’ multiplies a vector by a single
 * precision floating point scalar and adds another vector, ‘y = a*y +
 * x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_saypx(
    float a,
    struct mtxvector_packed * y,
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_daypx()’ multiplies a vector by a double
 * precision floating point scalar and adds another vector, ‘y = a*y +
 * x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_daypx(
    double a,
    struct mtxvector_packed * y,
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_sdot()’ cpackedutes the Euclidean dot product of
 * two vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_sdot(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_ddot()’ cpackedutes the Euclidean dot product of
 * two vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_ddot(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_cdotu()’ cpackedutes the product of the transpose
 * of a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_cdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_zdotu()’ cpackedutes the product of the transpose
 * of a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_zdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_cdotc()’ cpackedutes the Euclidean dot product of
 * two complex vectors in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_cdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_zdotc()’ cpackedutes the Euclidean dot product of
 * two complex vectors in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_zdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_snrm2()’ cpackedutes the Euclidean norm of a
 * vector in single precision floating point.
 */
int mtxvector_packed_snrm2(
    const struct mtxvector_packed * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_dnrm2()’ cpackedutes the Euclidean norm of a
 * vector in double precision floating point.
 */
int mtxvector_packed_dnrm2(
    const struct mtxvector_packed * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_sasum()’ cpackedutes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_packed_sasum(
    const struct mtxvector_packed * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_dasum()’ cpackedutes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_packed_dasum(
    const struct mtxvector_packed * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_packed_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_packed_iamax(
    const struct mtxvector_packed * x,
    int * iamax);

#endif
