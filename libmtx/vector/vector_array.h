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
 * Last modified: 2022-03-14
 *
 * Data structures for vectors in array format.
 */

#ifndef LIBMTX_VECTOR_ARRAY_H
#define LIBMTX_VECTOR_ARRAY_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxpartition;
struct mtxvector;

/**
 * ‘mtxvector_array’ represents a vector in array format.
 */
struct mtxvector_array
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
    int size;

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
    } data;
};

/*
 * Memory management
 */

/**
 * ‘mtxvector_array_free()’ frees storage allocated for a vector.
 */
void mtxvector_array_free(
    struct mtxvector_array * vector);

/**
 * ‘mtxvector_array_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_array_alloc_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/**
 * ‘mtxvector_array_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_array_init_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/*
 * Vector array formats
 */

/**
 * ‘mtxvector_array_alloc()’ allocates a vector in array format.
 */
int mtxvector_array_alloc(
    struct mtxvector_array * vector,
    enum mtxfield field,
    enum mtxprecision precision,
    int size);

/**
 * ‘mtxvector_array_init_real_single()’ allocates and initialises a
 * vector in array format with real, single precision coefficients.
 */
int mtxvector_array_init_real_single(
    struct mtxvector_array * vector,
    int size,
    const float * data);

/**
 * ‘mtxvector_array_init_real_double()’ allocates and initialises a
 * vector in array format with real, double precision coefficients.
 */
int mtxvector_array_init_real_double(
    struct mtxvector_array * vector,
    int size,
    const double * data);

/**
 * ‘mtxvector_array_init_complex_single()’ allocates and initialises a
 * vector in array format with complex, single precision coefficients.
 */
int mtxvector_array_init_complex_single(
    struct mtxvector_array * vector,
    int size,
    const float (* data)[2]);

/**
 * ‘mtxvector_array_init_complex_double()’ allocates and initialises a
 * vector in array format with complex, double precision coefficients.
 */
int mtxvector_array_init_complex_double(
    struct mtxvector_array * vector,
    int size,
    const double (* data)[2]);

/**
 * ‘mtxvector_array_init_integer_single()’ allocates and initialises a
 * vector in array format with integer, single precision coefficients.
 */
int mtxvector_array_init_integer_single(
    struct mtxvector_array * vector,
    int size,
    const int32_t * data);

/**
 * ‘mtxvector_array_init_integer_double()’ allocates and initialises a
 * vector in array format with integer, double precision coefficients.
 */
int mtxvector_array_init_integer_double(
    struct mtxvector_array * vector,
    int size,
    const int64_t * data);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_array_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_array_set_constant_real_single(
    struct mtxvector_array * vector,
    float a);

/**
 * ‘mtxvector_array_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_array_set_constant_real_double(
    struct mtxvector_array * vector,
    double a);

/**
 * ‘mtxvector_array_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_array_set_constant_complex_single(
    struct mtxvector_array * vector,
    float a[2]);

/**
 * ‘mtxvector_array_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_array_set_constant_complex_double(
    struct mtxvector_array * vector,
    double a[2]);

/**
 * ‘mtxvector_array_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_array_set_constant_integer_single(
    struct mtxvector_array * vector,
    int32_t a);

/**
 * ‘mtxvector_array_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_array_set_constant_integer_double(
    struct mtxvector_array * vector,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_array_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_array_from_mtxfile(
    struct mtxvector_array * vector,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxvector_array_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_array_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_array * vector,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_array_partition()’ partitions a vector into blocks
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
int mtxvector_array_partition(
    struct mtxvector * dsts,
    const struct mtxvector_array * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_array_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_array_join(
    struct mtxvector_array * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_array_swap()’ swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_swap(
    struct mtxvector_array * x,
    struct mtxvector_array * y);

/**
 * ‘mtxvector_array_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_copy(
    struct mtxvector_array * y,
    const struct mtxvector_array * x);

/**
 * ‘mtxvector_array_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_array_sscal(
    float a,
    struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_array_dscal(
    double a,
    struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_array_cscal(
    float a[2],
    struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_array_zscal(
    double a[2],
    struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_saxpy()’ adds a vector to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_saxpy(
    float a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_daxpy()’ adds a vector to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_daxpy(
    double a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_saypx(
    float a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_daypx(
    double a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_sdot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_ddot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_cdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_array_zdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_array_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_array_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_array_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_array_snrm2(
    const struct mtxvector_array * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_array_dnrm2(
    const struct mtxvector_array * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_array_sasum(
    const struct mtxvector_array * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_array_dasum(
    const struct mtxvector_array * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_array_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_array_iamax(
    const struct mtxvector_array * x,
    int * iamax);

/*
 * Sorting
 */

/**
 * ‘mtxvector_array_permute()’ permutes the elements of a vector
 * according to a given permutation.
 *
 * The array ‘perm’ should be an array of length ‘size’ that stores a
 * permutation of the integers ‘0,1,...,N-1’, where ‘N’ is the number
 * of vector elements.
 *
 * After permuting, the 1st vector element of the original vector is
 * now located at position ‘perm[0]’ in the sorted vector ‘x’, the 2nd
 * element is now at position ‘perm[1]’, and so on.
 */
int mtxvector_array_permute(
    struct mtxvector_array * x,
    int64_t offset,
    int64_t size,
    int64_t * perm);

/**
 * ‘mtxvector_array_sort()’ sorts elements of a vector by the given
 * keys.
 *
 * The array ‘keys’ must be an array of length ‘size’ that stores a
 * 64-bit unsigned integer sorting key that is used to define the
 * order in which to sort the vector elements..
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’, which is then used to store the sorting permutation. That
 * is, ‘perm’ is a permutation of the integers ‘0,1,...,N-1’, where
 * ‘N’ is the number of vector elements, such that the 1st vector
 * element in the original vector is now located at position ‘perm[0]’
 * in the sorted vector ‘x’, the 2nd element is now at position
 * ‘perm[1]’, and so on.
 */
int mtxvector_array_sort(
    struct mtxvector_array * x,
    int64_t size,
    uint64_t * keys,
    int64_t * perm);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_array_send()’ sends Matrix Market data lines to another
 * MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_array_recv()’.
 */
int mtxvector_array_send(
    const struct mtxvector_array * data,
    int64_t size,
    int64_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_recv()’ receives Matrix Market data lines from
 * another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_array_send()’.
 */
int mtxvector_array_recv(
    struct mtxvector_array * data,
    int64_t size,
    int64_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_bcast()’ broadcasts Matrix Market data lines from
 * an MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_array_bcast()’.
 */
int mtxvector_array_bcast(
    struct mtxvector_array * data,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_gatherv()’ gathers Matrix Market data lines onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_array_gatherv()’.
 */
int mtxvector_array_gatherv(
    const struct mtxvector_array * sendbuf,
    int64_t sendoffset,
    int sendcount,
    struct mtxvector_array * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_scatterv()’ scatters Matrix Market data lines from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_array_scatterv()’.
 */
int mtxvector_array_scatterv(
    const struct mtxvector_array * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    struct mtxvector_array * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_alltoallv()’ performs an all-to-all exchange of
 * Matrix Market data lines between MPI processes in a communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxvector_array_alltoallv()’.
 */
int mtxvector_array_alltoallv(
    const struct mtxvector_array * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    struct mtxvector_array * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
