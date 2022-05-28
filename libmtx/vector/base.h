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
 * Last modified: 2022-05-28
 *
 * Data structures and routines for basic dense vectors.
 */

#ifndef LIBMTX_VECTOR_BASE_H
#define LIBMTX_VECTOR_BASE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/vector/precision.h>
#include <libmtx/vector/field.h>
#include <libmtx/mtxfile/header.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxpartition;
struct mtxvector;
struct mtxvector_packed;

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
 * ‘mtxvector_base_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_base_init_pattern(
    struct mtxvector_base * x,
    int64_t size);

/*
 * initialise vectors from strided arrays
 */

/**
 * ‘mtxvector_base_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_base_init_strided_real_single(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const float * data);

/**
 * ‘mtxvector_base_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_base_init_strided_real_double(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const double * data);

/**
 * ‘mtxvector_base_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_base_init_strided_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2]);

/**
 * ‘mtxvector_base_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_base_init_strided_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2]);

/**
 * ‘mtxvector_base_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_base_init_strided_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const int32_t * data);

/**
 * ‘mtxvector_base_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_base_init_strided_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    int64_t stride,
    const int64_t * data);

/*
 * accessing values
 */

/**
 * ‘mtxvector_base_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_real_single(
    const struct mtxvector_base * x,
    int64_t size,
    int stride,
    float * a);

/**
 * ‘mtxvector_base_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_real_double(
    const struct mtxvector_base * x,
    int64_t size,
    int stride,
    double * a);

/**
 * ‘mtxvector_base_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    float (* a)[2]);

/**
 * ‘mtxvector_base_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    double (* a)[2]);

/**
 * ‘mtxvector_base_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    int32_t * a);

/**
 * ‘mtxvector_base_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_base_get_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    int64_t * a);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_base_setzero()’ sets every value of a vector to zero.
 */
int mtxvector_base_setzero(
    struct mtxvector_base * x);

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

/**
 * ‘mtxvector_base_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxvector_base_set_real_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const float * a);

/**
 * ‘mtxvector_base_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxvector_base_set_real_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const double * a);

/**
 * ‘mtxvector_base_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxvector_base_set_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const float (*a)[2]);

/**
 * ‘mtxvector_base_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxvector_base_set_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const double (*a)[2]);

/**
 * ‘mtxvector_base_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_base_set_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const int32_t * a);

/**
 * ‘mtxvector_base_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_base_set_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    int stride,
    const int64_t * a);

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
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_base_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxvector_base’. If successful, then ‘dsts[p]’
 * points to a vector consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxvector_base_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxvector_base_split(
    int num_parts,
    struct mtxvector_base ** dsts,
    const struct mtxvector_base * src,
    int64_t size,
    int * parts);

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
 * ‘mtxvector_base_sdot()’ computes the Euclidean dot product of two
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
 * ‘mtxvector_base_ddot()’ computes the Euclidean dot product of two
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
 * ‘mtxvector_base_cdotu()’ computes the product of the transpose of
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
 * ‘mtxvector_base_zdotu()’ computes the product of the transpose of
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
 * ‘mtxvector_base_cdotc()’ computes the Euclidean dot product of two
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
 * ‘mtxvector_base_zdotc()’ computes the Euclidean dot product of two
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
 * ‘mtxvector_base_snrm2()’ computes the Euclidean norm of a vector
 * in single precision floating point.
 */
int mtxvector_base_snrm2(
    const struct mtxvector_base * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_dnrm2()’ computes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxvector_base_dnrm2(
    const struct mtxvector_base * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_sasum(
    const struct mtxvector_base * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_dasum()’ computes the sum of absolute values
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

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_base_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_ussdot(
    const struct mtxvector_packed * x,
    const struct mtxvector_base * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_usddot(
    const struct mtxvector_packed * x,
    const struct mtxvector_base * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uscdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uszdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uscdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_base_uszdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_base_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_ussaxpy(
    struct mtxvector_base * y,
    float alpha,
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_usdaxpy(
    struct mtxvector_base * y,
    double alpha,
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_uscaxpy(
    struct mtxvector_base * y,
    float alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_base_uszaxpy(
    struct mtxvector_base * y,
    double alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_base_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxvector_base_usga(
    struct mtxvector_packed * x,
    const struct mtxvector_base * y);

/**
 * ‘mtxvector_base_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxvector_base_usgz(
    struct mtxvector_packed * x,
    struct mtxvector_base * y);

/**
 * ‘mtxvector_base_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxvector_base_ussc(
    struct mtxvector_base * y,
    const struct mtxvector_packed * x);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxvector_base_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxvector_base_usscga(
    struct mtxvector_packed * z,
    const struct mtxvector_packed * x);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_base_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_base_recv()’.
 */
int mtxvector_base_send(
    const struct mtxvector_base * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxvector_base_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_base_send()’.
 */
int mtxvector_base_recv(
    struct mtxvector_base * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode);

/**
 * ‘mtxvector_base_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_base_send()’.
 */
int mtxvector_base_irecv(
    struct mtxvector_base * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode);
#endif

#endif
