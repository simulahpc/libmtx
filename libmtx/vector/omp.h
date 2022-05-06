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
 * Data structures and routines for shared-memory parallel, dense
 * vectors using OpenMP.
 */

#ifndef LIBMTX_VECTOR_OMP_H
#define LIBMTX_VECTOR_OMP_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_OPENMP
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
     * ‘base’ is the underlying dense vector.
     */
    struct mtxvector_base base;
};

/*
 * Memory management
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
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_omp_alloc()’ allocates a vector.
 */
int mtxvector_omp_alloc(
    struct mtxvector_omp * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int num_threads);

/**
 * ‘mtxvector_omp_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_omp_init_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_omp_init_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    const double * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float (* data)[2],
    int num_threads);

/**
 * ‘mtxvector_omp_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    const double (* data)[2],
    int num_threads);

/**
 * ‘mtxvector_omp_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    const int32_t * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    const int64_t * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_omp_init_pattern(
    struct mtxvector_omp * x,
    int64_t size,
    int num_threads);

/*
 * initialise vectors from strided arrays
 */

/**
 * ‘mtxvector_omp_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_omp_init_strided_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_omp_init_strided_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const double * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_strided_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2],
    int num_threads);

/**
 * ‘mtxvector_omp_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_strided_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2],
    int num_threads);

/**
 * ‘mtxvector_omp_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_strided_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const int32_t * data,
    int num_threads);

/**
 * ‘mtxvector_omp_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_strided_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const int64_t * data,
    int num_threads);

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
 * ‘mtxvector_omp_partition()’ partitions a vector into blocks
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
int mtxvector_omp_partition(
    struct mtxvector * dsts,
    const struct mtxvector_omp * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_omp_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_omp_join(
    struct mtxvector_omp * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

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
    const struct mtxvector_packed * x,
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
    const struct mtxvector_packed * x,
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
    const struct mtxvector_packed * x,
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
    const struct mtxvector_packed * x,
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
    const struct mtxvector_packed * x,
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
    const struct mtxvector_packed * x,
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
    struct mtxvector_omp * y,
    float alpha,
    const struct mtxvector_packed * x,
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
    struct mtxvector_omp * y,
    double alpha,
    const struct mtxvector_packed * x,
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
    struct mtxvector_omp * y,
    float alpha[2],
    const struct mtxvector_packed * x,
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
    struct mtxvector_omp * y,
    double alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_omp_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxvector_omp_usga(
    struct mtxvector_packed * x,
    const struct mtxvector_omp * y);

/**
 * ‘mtxvector_omp_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxvector_omp_usgz(
    struct mtxvector_packed * xpacked,
    struct mtxvector_omp * y);

/**
 * ‘mtxvector_omp_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_omp_ussc(
    struct mtxvector_omp * y,
    const struct mtxvector_packed * x);

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
    struct mtxvector_packed * z,
    const struct mtxvector_packed * x);

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
