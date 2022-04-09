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
 * Data structures and routines for dense vectors with vector
 * operations accelerated by an external BLAS library.
 */

#ifndef LIBMTX_VECTOR_BLAS_H
#define LIBMTX_VECTOR_BLAS_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_BLAS
#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/vector/base.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxpartition;
struct mtxvector;
struct mtxvector_packed;

/**
 * ‘mtxvector_blas’ represents a dense vector that can perform
 * operations accelerated by an external BLAS library.
 */
struct mtxvector_blas
{
    /**
     * ‘base’ is the underlying dense vector.
     */
    struct mtxvector_base base;
};

/*
 * Memory management
 */

/**
 * ‘mtxvector_blas_free()’ frees storage allocated for a vector.
 */
void mtxvector_blas_free(
    struct mtxvector_blas * x);

/**
 * ‘mtxvector_blas_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_blas_alloc_copy(
    struct mtxvector_blas * dst,
    const struct mtxvector_blas * src);

/**
 * ‘mtxvector_blas_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_blas_init_copy(
    struct mtxvector_blas * dst,
    const struct mtxvector_blas * src);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_blas_alloc()’ allocates a vector.
 */
int mtxvector_blas_alloc(
    struct mtxvector_blas * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size);

/**
 * ‘mtxvector_blas_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_blas_init_real_single(
    struct mtxvector_blas * x,
    int64_t size,
    const float * data);

/**
 * ‘mtxvector_blas_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_blas_init_real_double(
    struct mtxvector_blas * x,
    int64_t size,
    const double * data);

/**
 * ‘mtxvector_blas_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_blas_init_complex_single(
    struct mtxvector_blas * x,
    int64_t size,
    const float (* data)[2]);

/**
 * ‘mtxvector_blas_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_blas_init_complex_double(
    struct mtxvector_blas * x,
    int64_t size,
    const double (* data)[2]);

/**
 * ‘mtxvector_blas_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_blas_init_integer_single(
    struct mtxvector_blas * x,
    int64_t size,
    const int32_t * data);

/**
 * ‘mtxvector_blas_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_blas_init_integer_double(
    struct mtxvector_blas * x,
    int64_t size,
    const int64_t * data);

/**
 * ‘mtxvector_blas_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_blas_init_pattern(
    struct mtxvector_blas * x,
    int64_t size);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_blas_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_blas_set_constant_real_single(
    struct mtxvector_blas * x,
    float a);

/**
 * ‘mtxvector_blas_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_blas_set_constant_real_double(
    struct mtxvector_blas * x,
    double a);

/**
 * ‘mtxvector_blas_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_blas_set_constant_complex_single(
    struct mtxvector_blas * x,
    float a[2]);

/**
 * ‘mtxvector_blas_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_blas_set_constant_complex_double(
    struct mtxvector_blas * x,
    double a[2]);

/**
 * ‘mtxvector_blas_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_blas_set_constant_integer_single(
    struct mtxvector_blas * x,
    int32_t a);

/**
 * ‘mtxvector_blas_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_blas_set_constant_integer_double(
    struct mtxvector_blas * x,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_blas_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_blas_from_mtxfile(
    struct mtxvector_blas * x,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxvector_blas_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_blas_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_blas * x,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_blas_partition()’ partitions a vector into blocks
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
int mtxvector_blas_partition(
    struct mtxvector * dsts,
    const struct mtxvector_blas * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_blas_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_blas_join(
    struct mtxvector_blas * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_blas_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_swap(
    struct mtxvector_blas * x,
    struct mtxvector_blas * y);

/**
 * ‘mtxvector_blas_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_copy(
    struct mtxvector_blas * y,
    const struct mtxvector_blas * x);

/**
 * ‘mtxvector_blas_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_blas_sscal(
    float a,
    struct mtxvector_blas * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_blas_dscal(
    double a,
    struct mtxvector_blas * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_blas_cscal(
    float a[2],
    struct mtxvector_blas * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_blas_zscal(
    double a[2],
    struct mtxvector_blas * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_saxpy(
    float a,
    const struct mtxvector_blas * x,
    struct mtxvector_blas * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_daxpy(
    double a,
    const struct mtxvector_blas * x,
    struct mtxvector_blas * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_saypx(
    float a,
    struct mtxvector_blas * y,
    const struct mtxvector_blas * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_daypx(
    double a,
    struct mtxvector_blas * y,
    const struct mtxvector_blas * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_sdot(
    const struct mtxvector_blas * x,
    const struct mtxvector_blas * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_ddot(
    const struct mtxvector_blas * x,
    const struct mtxvector_blas * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_cdotu(
    const struct mtxvector_blas * x,
    const struct mtxvector_blas * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_zdotu(
    const struct mtxvector_blas * x,
    const struct mtxvector_blas * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_cdotc(
    const struct mtxvector_blas * x,
    const struct mtxvector_blas * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_zdotc(
    const struct mtxvector_blas * x,
    const struct mtxvector_blas * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_blas_snrm2(
    const struct mtxvector_blas * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_blas_dnrm2(
    const struct mtxvector_blas * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_blas_sasum(
    const struct mtxvector_blas * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_blas_dasum(
    const struct mtxvector_blas * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_blas_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_blas_iamax(
    const struct mtxvector_blas * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_blas_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed storage format.
 */
int mtxvector_blas_usga(
    struct mtxvector_packed * xpacked,
    const struct mtxvector_blas * yblas);

/**
 * ‘mtxvector_blas_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed storage format.
 */
int mtxvector_blas_ussc(
    struct mtxvector_blas * y,
    const struct mtxvector_packed * x);
#endif

#endif
