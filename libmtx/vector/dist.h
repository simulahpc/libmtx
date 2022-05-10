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
 * Last modified: 2022-04-14
 *
 * Data structures and routines for distributed sparse vectors in
 * packed form.
 */

#ifndef LIBMTX_VECTOR_DIST_H
#define LIBMTX_VECTOR_DIST_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

#include <mpi.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxdistfile2;
struct mtxdisterror;

/**
 * ‘mtxvector_dist’ represents a distributed sparse vector in packed
 * form.
 *
 * The vector is thus represented on each process by a contiguous
 * array of elements together with an array of integers designating
 * the offset of each element. This can be thought of as a sum of
 * sparse vectors in packed form with one vector per process.
 */
struct mtxvector_dist
{
    /**
     * ‘comm’ is an MPI communicator for processes among which the
     * vector is distributed.
     */
    MPI_Comm comm;

    /**
     * ‘comm_size’ is the size of the MPI communicator. This is equal
     * to the number of parts in the partitioning of the vector.
     */
    int comm_size;

    /**
     * ‘rank’ is the rank of the current process.
     */
    int rank;

    /**
     * ‘size’ is the number of vector elements, which must be the same
     * across all processes in the communicator ‘comm’, and it must
     * also be equal to ‘xp.size’.
     */
    int64_t size;

    /**
     * ‘num_nonzeros’ is the total number of explicitly stored vector
     * entries for the distributed sparse vector in packed form. This
     * is equal to the sum of the number of explicitly stored vector
     * entries on each process (‘xp.num_nonzeros’).
     */
    int64_t num_nonzeros;

    /**
     * ‘xp’ is the underlying storage of the sparse vector in packed
     * form belonging to the current process.
     */
    struct mtxvector_packed xp;

    /**
     * ‘blksize’ is the number of vector elements owned by the current
     * process, if the elements were partitioned and distributed among
     * processes in equal-sized blocks.
     */
    int64_t blksize;

    /**
     * ‘blkstart’ is the offset to the first vector element owned by
     * the current process, if the elements were partitioned and
     * distributed among processes in equal-sized, contiguous blocks.
     */
    int64_t blkstart;

    /**
     * ‘ranks’ is an array of size ‘blksize’. The “assumed partition”
     * strategy assumes that elements are partitioned and distributed
     * among processes in equal-sized, contiguous blocks. For each
     * element in the block of the current process, the array ‘ranks’
     * contains the rank of the process that actually owns the
     * element.
     */
    int * ranks;

    /**
     * ‘localidx’ is an array of size ‘blksize’. The “assumed
     * partition” strategy assumes that elements are partitioned and
     * distributed among processes in equal-sized, contiguous
     * blocks. For each element in the block of the current process,
     * the array ‘localidx’ contains the offset of that element
     * according to the local numbering of elements on the process
     * that actually owns the element.
     */
    int64_t * localidx;
};

/*
 * Memory management
 */

/**
 * ‘mtxvector_dist_free()’ frees storage allocated for a vector.
 */
void mtxvector_dist_free(
    struct mtxvector_dist * x);

/**
 * ‘mtxvector_dist_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_dist_alloc_copy(
    struct mtxvector_dist * dst,
    const struct mtxvector_dist * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_dist_init_copy(
    struct mtxvector_dist * dst,
    const struct mtxvector_dist * src,
    struct mtxdisterror * disterr);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_dist_alloc()’ allocates a sparse vector in packed form,
 * where nonzero coefficients are stored in an underlying dense vector
 * of the given type.
 */
int mtxvector_dist_alloc(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 *
 * On each process, ‘idx’ and ‘data’ are arrays of length
 * ‘num_nonzeros’, containing the global offsets and values,
 * respectively, of the vector elements stored on the process. Note
 * that ‘num_nonzeros’ may differ from one process to the next.
 */
int mtxvector_dist_init_real_single(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_dist_init_real_double(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_dist_init_complex_single(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_dist_init_complex_double(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_dist_init_integer_single(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_dist_init_integer_double(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_pattern()’ allocates and initialises a binary
 * pattern vector, where every entry has a value of one.
 */
int mtxvector_dist_init_pattern(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_dist_init_strided_real_single(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_dist_init_strided_real_double(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_dist_init_strided_complex_single(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_dist_init_strided_complex_double(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_dist_init_strided_integer_single(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_dist_init_strided_integer_double(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_init_pattern()’ allocates and initialises a binary
 * pattern vector, where every entry has a value of one.
 */
int mtxvector_dist_init_strided_pattern(
    struct mtxvector_dist * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Modifying values
 */

/**
 * ‘mtxvector_dist_setzero()’ sets every nonzero entry of a vector to
 * zero.
 */
int mtxvector_dist_setzero(
    struct mtxvector_dist * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_set_constant_real_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point number.
 */
int mtxvector_dist_set_constant_real_single(
    struct mtxvector_dist * x,
    float a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_set_constant_real_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point number.
 */
int mtxvector_dist_set_constant_real_double(
    struct mtxvector_dist * x,
    double a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_set_constant_complex_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxvector_dist_set_constant_complex_single(
    struct mtxvector_dist * x,
    float a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_set_constant_complex_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxvector_dist_set_constant_complex_double(
    struct mtxvector_dist * x,
    double a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_set_constant_integer_single()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxvector_dist_set_constant_integer_single(
    struct mtxvector_dist * x,
    int32_t a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_set_constant_integer_double()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxvector_dist_set_constant_integer_double(
    struct mtxvector_dist * x,
    int64_t a,
    struct mtxdisterror * disterr);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_dist_from_mtxfile()’ converts from a vector in Matrix
 * Market format.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each process.
 */
int mtxvector_dist_from_mtxfile(
    struct mtxvector_dist * x,
    const struct mtxfile * mtxfile,
    enum mtxvectortype type,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_to_mtxfile()’ converts to a vector in Matrix Market
 * format.
 */
int mtxvector_dist_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_dist * x,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_from_mtxdistfile2()’ converts from a vector in
 * Matrix Market format that is distributed among multiple processes.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each process.
 */
int mtxvector_dist_from_mtxdistfile2(
    struct mtxvector_dist * x,
    const struct mtxdistfile2 * mtxdistfile2,
    enum mtxvectortype type,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_to_mtxdistfile2()’ converts to a vector in Matrix
 * Market format that is distributed among multiple processes.
 */
int mtxvector_dist_to_mtxdistfile2(
    struct mtxdistfile2 * mtxdistfile2,
    const struct mtxvector_dist * x,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr);

/*
 * I/O operations
 */

/**
 * ‘mtxvector_dist_fwrite()’ writes a distributed vector to a single
 * stream that is shared by every process in the communicator. The
 * output is written in Matrix Market format.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed data to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxvector_dist_fwrite(
    const struct mtxvector_dist * x,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_dist_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both vectors must also have the same number of
 * nonzero elements on that process.
 */
int mtxvector_dist_swap(
    struct mtxvector_dist * x,
    struct mtxvector_dist * y,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both vectors must also have the same number of
 * nonzero elements on that process.
 */
int mtxvector_dist_copy(
    struct mtxvector_dist * y,
    const struct mtxvector_dist * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_dist_sscal(
    float a,
    struct mtxvector_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_dist_dscal(
    double a,
    struct mtxvector_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_dist_cscal(
    float a[2],
    struct mtxvector_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_dist_zscal(
    double a[2],
    struct mtxvector_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the dist vectors are
 * allowed.
 */
int mtxvector_dist_saxpy(
    float a,
    const struct mtxvector_dist * x,
    struct mtxvector_dist * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the dist vectors are
 * allowed.
 */
int mtxvector_dist_daxpy(
    double a,
    const struct mtxvector_dist * x,
    struct mtxvector_dist * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the dist vectors are
 * allowed.
 */
int mtxvector_dist_saypx(
    float a,
    struct mtxvector_dist * y,
    const struct mtxvector_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the dist vectors are
 * allowed.
 */
int mtxvector_dist_daypx(
    double a,
    struct mtxvector_dist * y,
    const struct mtxvector_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxvector_dist_sdot(
    const struct mtxvector_dist * x,
    const struct mtxvector_dist * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxvector_dist_ddot(
    const struct mtxvector_dist * x,
    const struct mtxvector_dist * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxvector_dist_cdotu(
    const struct mtxvector_dist * x,
    const struct mtxvector_dist * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxvector_dist_zdotu(
    const struct mtxvector_dist * x,
    const struct mtxvector_dist * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxvector_dist_cdotc(
    const struct mtxvector_dist * x,
    const struct mtxvector_dist * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxvector_dist_zdotc(
    const struct mtxvector_dist * x,
    const struct mtxvector_dist * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point. Repeated indices in the dist
 * vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_dist_snrm2(
    const struct mtxvector_dist * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point. Repeated indices in the dist
 * vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_dist_dnrm2(
    const struct mtxvector_dist * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_dist_sasum(
    const struct mtxvector_dist * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_dist_dasum(
    const struct mtxvector_dist * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_dist_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts. Repeated indices in the dist vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_dist_iamax(
    const struct mtxvector_dist * x,
    int * iamax,
    struct mtxdisterror * disterr);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxvector_dist_usscga()’ performs a combined scatter-gather
 * operation from a distributed sparse vector ‘x’ in packed form into
 * another distributed sparse vector ‘z’ in packed form. Repeated
 * indices in the packed vector ‘x’ are not allowed, otherwise the
 * result is undefined. They are, however, allowed in the packed
 * vector ‘z’.
 */
int mtxvector_dist_usscga(
    struct mtxvector_dist * z,
    const struct mtxvector_dist * x,
    struct mtxdisterror * disterr);
#endif
#endif
