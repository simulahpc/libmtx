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
 * Last modified: 2022-03-22
 *
 * Data structures for vectors in coordinate format.
 */

#ifndef LIBMTX_VECTOR_COORDINATE_H
#define LIBMTX_VECTOR_COORDINATE_H

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
 * ‘mtxvector_coordinate’ represents a vector in coordinate format.
 */
struct mtxvector_coordinate
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
     * ‘num_entries’ is the number of vector elements.
     */
    int64_t num_entries;

    /**
     * ‘num_nonzeros’ is the number of nonzero vector entries for a
     * sparse vector.
     */
    int64_t num_nonzeros;

    /**
     * ‘size’ is the number of explicitly stored vector entries of a
     * coordinate vector, which is the same as ‘num_nonzeros’.
     */
    int64_t size;

    /**
     * ‘indices’ is an array containing the locations of nonzero
     * vector entries.  Note that indices are 0-based, unlike the
     * Matrix Market format, where indices are 1-based.
     */
    int * indices;

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
 * ‘mtxvector_coordinate_free()’ frees storage allocated for a vector.
 */
void mtxvector_coordinate_free(
    struct mtxvector_coordinate * vector);

/**
 * ‘mtxvector_coordinate_alloc_copy()’ allocates a copy of a vector
 * without initialising the values.
 */
int mtxvector_coordinate_alloc_copy(
    struct mtxvector_coordinate * dst,
    const struct mtxvector_coordinate * src);

/**
 * ‘mtxvector_coordinate_init_copy()’ allocates a copy of a vector and
 * also copies the values.
 */
int mtxvector_coordinate_init_copy(
    struct mtxvector_coordinate * dst,
    const struct mtxvector_coordinate * src);

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxvector_coordinate_alloc()’ allocates a vector in coordinate
 * format.
 */
int mtxvector_coordinate_alloc(
    struct mtxvector_coordinate * vector,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t num_entries,
    int64_t num_nonzeros);

/**
 * ‘mtxvector_coordinate_init_real_single()’ allocates and initialises
 * a vector in coordinate format with real, single precision
 * coefficients.
 */
int mtxvector_coordinate_init_real_single(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices,
    const float * data);

/**
 * ‘mtxvector_coordinate_init_real_double()’ allocates and initialises
 * a vector in coordinate format with real, double precision
 * coefficients.
 */
int mtxvector_coordinate_init_real_double(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices,
    const double * data);

/**
 * ‘mtxvector_coordinate_init_complex_single()’ allocates and
 * initialises a vector in coordinate format with complex, single
 * precision coefficients.
 */
int mtxvector_coordinate_init_complex_single(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices,
    const float (* data)[2]);

/**
 * ‘mtxvector_coordinate_init_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxvector_coordinate_init_complex_double(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices,
    const double (* data)[2]);

/**
 * ‘mtxvector_coordinate_init_integer_single()’ allocates and
 * initialises a vector in coordinate format with integer, single
 * precision coefficients.
 */
int mtxvector_coordinate_init_integer_single(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices,
    const int32_t * data);

/**
 * ‘mtxvector_coordinate_init_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxvector_coordinate_init_integer_double(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices,
    const int64_t * data);

/**
 * ‘mtxvector_coordinate_init_pattern()’ allocates and initialises a
 * vector in coordinate format with boolean coefficients.
 */
int mtxvector_coordinate_init_pattern(
    struct mtxvector_coordinate * vector,
    int64_t num_entries,
    int64_t num_nonzeros,
    const int * indices);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_coordinate_set_constant_real_single()’ sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point number.
 */
int mtxvector_coordinate_set_constant_real_single(
    struct mtxvector_coordinate * vector,
    float a);

/**
 * ‘mtxvector_coordinate_set_constant_real_double()’ sets every
 * nonzero value of a vector equal to a constant, double precision
 * floating point number.
 */
int mtxvector_coordinate_set_constant_real_double(
    struct mtxvector_coordinate * vector,
    double a);

/**
 * ‘mtxvector_coordinate_set_constant_complex_single()’ sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxvector_coordinate_set_constant_complex_single(
    struct mtxvector_coordinate * vector,
    float a[2]);

/**
 * ‘mtxvector_coordinate_set_constant_complex_double()’ sets every
 * nonzero value of a vector equal to a constant, double precision
 * floating point complex number.
 */
int mtxvector_coordinate_set_constant_complex_double(
    struct mtxvector_coordinate * vector,
    double a[2]);

/**
 * ‘mtxvector_coordinate_set_constant_integer_single()’ sets every
 * nonzero value of a vector equal to a constant integer.
 */
int mtxvector_coordinate_set_constant_integer_single(
    struct mtxvector_coordinate * vector,
    int32_t a);

/**
 * ‘mtxvector_coordinate_set_constant_integer_double()’ sets every
 * nonzero value of a vector equal to a constant integer.
 */
int mtxvector_coordinate_set_constant_integer_double(
    struct mtxvector_coordinate * vector,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_coordinate_from_mtxfile()’ converts a vector in Matrix
 * Market format to a vector in coordinate format.
 */
int mtxvector_coordinate_from_mtxfile(
    struct mtxvector_coordinate * vector,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxvector_coordinate_to_mtxfile()’ converts a vector in coordinate
 * format to a vector in Matrix Market format.
 */
int mtxvector_coordinate_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_coordinate * vector,
    enum mtxfileformat mtxfmt);

/*
 * Partitioning
 */

/**
 * ‘mtxvector_coordinate_partition()’ partitions a vector into blocks
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
int mtxvector_coordinate_partition(
    struct mtxvector * dsts,
    const struct mtxvector_coordinate * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_coordinate_join()’ joins together block vectors to form
 * a larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_coordinate_join(
    struct mtxvector_coordinate * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_coordinate_swap()’ swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_swap(
    struct mtxvector_coordinate * x,
    struct mtxvector_coordinate * y);

/**
 * ‘mtxvector_coordinate_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_copy(
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x);

/**
 * ‘mtxvector_coordinate_sscal()’ scales a vector by a single
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxvector_coordinate_sscal(
    float a,
    struct mtxvector_coordinate * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_dscal()’ scales a vector by a double
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxvector_coordinate_dscal(
    double a,
    struct mtxvector_coordinate * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_coordinate_cscal(
    float a[2],
    struct mtxvector_coordinate * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_coordinate_zscal(
    double a[2],
    struct mtxvector_coordinate * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_saxpy()’ adds a vector to another vector
 * multiplied by a single precision floating point value, ‘y = a*x+y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_saxpy(
    float a,
    const struct mtxvector * x,
    struct mtxvector_coordinate * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_daxpy()’ adds a vector to another vector
 * multiplied by a double precision floating point value, ‘y = a*x+y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_daxpy(
    double a,
    const struct mtxvector * x,
    struct mtxvector_coordinate * y,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_saypx()’ multiplies a vector by a single
 * precision floating point scalar and adds another vector, ‘y=a*y+x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_saypx(
    float a,
    struct mtxvector_coordinate * y,
    const struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_daypx()’ multiplies a vector by a double
 * precision floating point scalar and adds another vector, ‘y=a*y+x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_daypx(
    double a,
    struct mtxvector_coordinate * y,
    const struct mtxvector * x,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_sdot()’ computes the Euclidean dot product of
 * two vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_sdot(
    const struct mtxvector_coordinate * x,
    const struct mtxvector * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_ddot()’ computes the Euclidean dot product of
 * two vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_ddot(
    const struct mtxvector_coordinate * x,
    const struct mtxvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_cdotu()’ computes the product of the
 * transpose of a complex row vector with another complex row vector
 * in single precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_cdotu(
    const struct mtxvector_coordinate * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_zdotu()’ computes the product of the
 * transpose of a complex row vector with another complex row vector
 * in double precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_zdotu(
    const struct mtxvector_coordinate * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_cdotc()’ computes the Euclidean dot product
 * of two complex vectors in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_cdotc(
    const struct mtxvector_coordinate * x,
    const struct mtxvector * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_zdotc()’ computes the Euclidean dot product
 * of two complex vectors in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_zdotc(
    const struct mtxvector_coordinate * x,
    const struct mtxvector * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_snrm2()’ computes the Euclidean norm of a
 * vector in single precision floating point.
 */
int mtxvector_coordinate_snrm2(
    const struct mtxvector_coordinate * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_dnrm2()’ computes the Euclidean norm of a
 * vector in double precision floating point.
 */
int mtxvector_coordinate_dnrm2(
    const struct mtxvector_coordinate * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_coordinate_sasum(
    const struct mtxvector_coordinate * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_coordinate_dasum(
    const struct mtxvector_coordinate * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxvector_coordinate_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_coordinate_iamax(
    const struct mtxvector_coordinate * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, "An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum," ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_coordinate_usga()’ performs a (sparse) gather from a
 * vector ‘y’ into another vector ‘x’.
 */
int mtxvector_coordinate_usga(
    const struct mtxvector * y,
    struct mtxvector_coordinate * x);

/**
 * ‘mtxvector_coordinate_usgz()’ performs a (sparse) gather from a
 * vector ‘y’ into another vector ‘x’, while zeroing the corresponding
 * elements of ‘y’ that were copied to ‘x’.
 */
int mtxvector_coordinate_usgz(
    const struct mtxvector * y,
    struct mtxvector_coordinate * x);

/**
 * ‘mtxvector_coordinate_ussc()’ performs a (sparse) scatter from a
 * vector ‘x’ into another vector ‘y’.
 */
int mtxvector_coordinate_ussc(
    const struct mtxvector * x,
    struct mtxvector_coordinate * y);

/*
 * Sorting
 */

/**
 * ‘mtxvector_coordinate_permute()’ permutes the elements of a vector
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
int mtxvector_coordinate_permute(
    struct mtxvector_coordinate * x,
    int64_t offset,
    int64_t size,
    int64_t * perm);

/**
 * ‘mtxvector_coordinate_sort()’ sorts elements of a vector by the
 * given keys.
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
int mtxvector_coordinate_sort(
    struct mtxvector_coordinate * x,
    int64_t size,
    uint64_t * keys,
    int64_t * perm);

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_coordinate_send()’ sends a vector to another MPI
 * process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to
 * ‘mtxvector_coordinate_recv()’.
 */
int mtxvector_coordinate_send(
    const struct mtxvector_coordinate * data,
    int64_t size,
    int64_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_recv()’ receives a vector from another MPI
 * process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_coordinate_send()’.
 */
int mtxvector_coordinate_recv(
    struct mtxvector_coordinate * data,
    int64_t size,
    int64_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_bcast()’ broadcasts a vector from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_coordinate_bcast()’.
 */
int mtxvector_coordinate_bcast(
    struct mtxvector_coordinate * data,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_gatherv()’ gathers a vector onto an MPI root
 * process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_coordinate_gatherv()’.
 */
int mtxvector_coordinate_gatherv(
    const struct mtxvector_coordinate * sendbuf,
    int64_t sendoffset,
    int sendcount,
    struct mtxvector_coordinate * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_scatterv()’ scatters a vector from an MPI
 * root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_coordinate_scatterv()’.
 */
int mtxvector_coordinate_scatterv(
    const struct mtxvector_coordinate * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    struct mtxvector_coordinate * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_alltoallv()’ performs an all-to-all exchange
 * of a vector between MPI processes in a communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxvector_coordinate_alltoallv()’.
 */
int mtxvector_coordinate_alltoallv(
    const struct mtxvector_coordinate * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    struct mtxvector_coordinate * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr);
#endif

#endif
