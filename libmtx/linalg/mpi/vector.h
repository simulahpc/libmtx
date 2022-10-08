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
 * Last modified: 2022-10-08
 *
 * Data structures and routines for distributed sparse vectors in
 * packed form.
 */

#ifndef LIBMTX_LINALG_MPI_VECTOR_H
#define LIBMTX_LINALG_MPI_VECTOR_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/linalg/local/vector.h>

#include <mpi.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxdistfile;
struct mtxdisterror;

/**
 * ‘mtxmpivector’ represents a distributed sparse vector in packed
 * form.
 *
 * The vector is thus represented on each process by a contiguous
 * array of elements together with an array of integers designating
 * the offset of each element. This can be thought of as a sum of
 * sparse vectors in packed form with one vector per process.
 */
struct mtxmpivector
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
    struct mtxvector xp;
};

/*
 * Memory management
 */

/**
 * ‘mtxmpivector_free()’ frees storage allocated for a vector.
 */
LIBMTX_API void mtxmpivector_free(
    struct mtxmpivector * x);

/**
 * ‘mtxmpivector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
LIBMTX_API int mtxmpivector_alloc_copy(
    struct mtxmpivector * dst,
    const struct mtxmpivector * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
LIBMTX_API int mtxmpivector_init_copy(
    struct mtxmpivector * dst,
    const struct mtxmpivector * src,
    struct mtxdisterror * disterr);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxmpivector_alloc()’ allocates a sparse vector in packed form,
 * where nonzero coefficients are stored in an underlying dense vector
 * of the given type.
 */
LIBMTX_API int mtxmpivector_alloc(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 *
 * On each process, ‘idx’ and ‘data’ are arrays of length
 * ‘num_nonzeros’, containing the global offsets and values,
 * respectively, of the vector elements stored on the process. Note
 * that ‘num_nonzeros’ may differ from one process to the next.
 */
LIBMTX_API int mtxmpivector_init_real_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_real_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_complex_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_complex_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_integer_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_integer_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_pattern()’ allocates and initialises a binary
 * pattern vector, where every entry has a value of one.
 */
LIBMTX_API int mtxmpivector_init_pattern(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_strided_real_single(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_strided_real_double(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_strided_complex_single(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_strided_complex_double(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_strided_integer_single(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
LIBMTX_API int mtxmpivector_init_strided_integer_double(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_init_pattern()’ allocates and initialises a binary
 * pattern vector, where every entry has a value of one.
 */
LIBMTX_API int mtxmpivector_init_strided_pattern(
    struct mtxmpivector * x,
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
 * ‘mtxmpivector_setzero()’ sets every nonzero entry of a vector to
 * zero.
 */
LIBMTX_API int mtxmpivector_setzero(
    struct mtxmpivector * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_set_constant_real_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point number.
 */
LIBMTX_API int mtxmpivector_set_constant_real_single(
    struct mtxmpivector * x,
    float a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_set_constant_real_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point number.
 */
LIBMTX_API int mtxmpivector_set_constant_real_double(
    struct mtxmpivector * x,
    double a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_set_constant_complex_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point complex number.
 */
LIBMTX_API int mtxmpivector_set_constant_complex_single(
    struct mtxmpivector * x,
    float a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_set_constant_complex_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point complex number.
 */
LIBMTX_API int mtxmpivector_set_constant_complex_double(
    struct mtxmpivector * x,
    double a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_set_constant_integer_single()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
LIBMTX_API int mtxmpivector_set_constant_integer_single(
    struct mtxmpivector * x,
    int32_t a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_set_constant_integer_double()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
LIBMTX_API int mtxmpivector_set_constant_integer_double(
    struct mtxmpivector * x,
    int64_t a,
    struct mtxdisterror * disterr);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmpivector_from_mtxfile()’ converts from a vector in Matrix
 * Market format.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each process.
 */
LIBMTX_API int mtxmpivector_from_mtxfile(
    struct mtxmpivector * x,
    const struct mtxfile * mtxfile,
    enum mtxvectortype type,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_to_mtxfile()’ converts to a vector in Matrix Market
 * format.
 */
LIBMTX_API int mtxmpivector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmpivector * x,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_from_mtxdistfile()’ converts from a vector in
 * Matrix Market format that is distributed among multiple processes.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each process.
 */
LIBMTX_API int mtxmpivector_from_mtxdistfile(
    struct mtxmpivector * x,
    const struct mtxdistfile * mtxdistfile,
    enum mtxvectortype type,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_to_mtxdistfile()’ converts to a vector in Matrix
 * Market format that is distributed among multiple processes.
 */
LIBMTX_API int mtxmpivector_to_mtxdistfile(
    struct mtxdistfile * mtxdistfile,
    const struct mtxmpivector * x,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr);

/*
 * I/O operations
 */

/**
 * ‘mtxmpivector_fwrite()’ writes a distributed vector to a single
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
LIBMTX_API int mtxmpivector_fwrite(
    const struct mtxmpivector * x,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr);

/*
 * partitioning
 */

/**
 * ‘mtxmpivector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxmpivector’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxmpivector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
LIBMTX_API int mtxmpivector_split(
    int num_parts,
    struct mtxmpivector ** dsts,
    const struct mtxmpivector * src,
    int64_t size,
    int * parts,
    int64_t * invperm,
    struct mtxdisterror * disterr);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmpivector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both vectors must also have the same number of
 * nonzero elements on that process.
 */
LIBMTX_API int mtxmpivector_swap(
    struct mtxmpivector * x,
    struct mtxmpivector * y,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both vectors must also have the same number of
 * nonzero elements on that process.
 */
LIBMTX_API int mtxmpivector_copy(
    struct mtxmpivector * y,
    const struct mtxmpivector * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
LIBMTX_API int mtxmpivector_sscal(
    float a,
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
LIBMTX_API int mtxmpivector_dscal(
    double a,
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
LIBMTX_API int mtxmpivector_cscal(
    float a[2],
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
LIBMTX_API int mtxmpivector_zscal(
    double a[2],
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_saxpy()’ adds a vector to another one multiplied by
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
LIBMTX_API int mtxmpivector_saxpy(
    float a,
    const struct mtxmpivector * x,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_daxpy()’ adds a vector to another one multiplied by
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
LIBMTX_API int mtxmpivector_daxpy(
    double a,
    const struct mtxmpivector * x,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_saypx()’ multiplies a vector by a single precision
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
LIBMTX_API int mtxmpivector_saypx(
    float a,
    struct mtxmpivector * y,
    const struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_daypx()’ multiplies a vector by a double precision
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
LIBMTX_API int mtxmpivector_daypx(
    double a,
    struct mtxmpivector * y,
    const struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_sdot()’ computes the Euclidean dot product of two
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
LIBMTX_API int mtxmpivector_sdot(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_ddot()’ computes the Euclidean dot product of two
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
LIBMTX_API int mtxmpivector_ddot(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_cdotu()’ computes the product of the transpose of a
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
LIBMTX_API int mtxmpivector_cdotu(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_zdotu()’ computes the product of the transpose of a
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
LIBMTX_API int mtxmpivector_zdotu(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_cdotc()’ computes the Euclidean dot product of two
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
LIBMTX_API int mtxmpivector_cdotc(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_zdotc()’ computes the Euclidean dot product of two
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
LIBMTX_API int mtxmpivector_zdotc(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point. Repeated indices in the dist
 * vector are not allowed, otherwise the result is undefined.
 */
LIBMTX_API int mtxmpivector_snrm2(
    const struct mtxmpivector * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point. Repeated indices in the dist
 * vector are not allowed, otherwise the result is undefined.
 */
LIBMTX_API int mtxmpivector_dnrm2(
    const struct mtxmpivector * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist vector are not allowed, otherwise the result is undefined.
 */
LIBMTX_API int mtxmpivector_sasum(
    const struct mtxmpivector * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist vector are not allowed, otherwise the result is undefined.
 */
LIBMTX_API int mtxmpivector_dasum(
    const struct mtxmpivector * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts. Repeated indices in the dist vector are not allowed,
 * otherwise the result is undefined.
 */
LIBMTX_API int mtxmpivector_iamax(
    const struct mtxmpivector * x,
    int * iamax,
    struct mtxdisterror * disterr);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxmpivector_usscga()’ performs a combined scatter-gather
 * operation from a distributed sparse vector ‘x’ in packed form into
 * another distributed sparse vector ‘z’ in packed form. Repeated
 * indices in the packed vector ‘x’ are not allowed, otherwise the
 * result is undefined. They are, however, allowed in the packed
 * vector ‘z’.
 */
int mtxmpivector_usscga(
    struct mtxmpivector * z,
    const struct mtxmpivector * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_usscga’ is a data structure for a persistent,
 * asynchronous, combined scatter-gather operation.
 */
struct mtxmpivector_usscga
{
    /**
     * ‘z’ is a distributed, sparse destination vector in packed form.
     */
    struct mtxmpivector * z;

    /**
     * ‘x’ is a distributed, sparse source vector in packed form.
     */
    const struct mtxmpivector * x;

    struct mtxmpivector_usscga_impl * impl;
};

/**
 * ‘mtxmpivector_usscga_init()’ allocates data structures for a
 * persistent, combined scatter-gather operation.
 *
 * This is used in cases where the combined scatter-gather operation
 * is performed repeatedly, since the setup phase only needs to be
 * carried out once.
 */
int mtxmpivector_usscga_init(
    struct mtxmpivector_usscga * usscga,
    struct mtxmpivector * z,
    const struct mtxmpivector * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_usscga_free()’ frees resources associated with a
 * persistent, combined scatter-gather operation.
 */
void mtxmpivector_usscga_free(
    struct mtxmpivector_usscga * usscga);

/**
 * ‘mtxmpivector_usscga_start()’ initiates a combined scatter-gather
 * operation from a distributed sparse vector ‘x’ in packed form into
 * another distributed sparse vector ‘z’ in packed form. Repeated
 * indices in the packed vector ‘x’ are not allowed, otherwise the
 * result is undefined. They are, however, allowed in the packed
 * vector ‘z’.
 *
 * The operation may not complete before
 * ‘mtxmpivector_usscga_wait()’ is called.
 */
int mtxmpivector_usscga_start(
    struct mtxmpivector_usscga * usscga,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_usscga_wait()’ waits for a persistent, combined
 * scatter-gather operation to finish.
 */
int mtxmpivector_usscga_wait(
    struct mtxmpivector_usscga * usscga,
    struct mtxdisterror * disterr);
#endif
#endif
