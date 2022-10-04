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
 * Last modified: 2022-06-06
 *
 * Data structures and routines for distributed matrices.
 */

#ifndef LIBMTX_MATRIX_DIST_H
#define LIBMTX_MATRIX_DIST_H

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/gemvoverlap.h>
#include <libmtx/linalg/transpose.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/precision.h>

#include <mpi.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxdistfile;
struct mtxdisterror;
struct mtxmpivector;

/**
 * ‘mtxmatrix_dist’ represents a distributed matrix.
 *
 * The matrix is thus represented on each process by a contiguous
 * array of elements together with an array of integers designating
 * the offset of each element. This can be thought of as a sum of
 * sparse matrices in packed form with one matrix per process.
 */
struct mtxmatrix_dist
{
    /**
     * ‘comm’ is an MPI communicator for processes among which the
     * matrix is distributed.
     */
    MPI_Comm comm;

    /**
     * ‘comm_size’ is the size of the MPI communicator. This is equal
     * to the number of parts in the partitioning of the matrix.
     */
    int comm_size;

    /**
     * ‘rank’ is the rank of the current process.
     */
    int rank;

    /**
     * ‘num_rows’ is the number of matrix rows, which must be the same
     * across all processes in the communicator ‘comm’
     */
    int64_t num_rows;

    /**
     * ‘num_columns’ is the number of matrix columns, which must be
     * the same across all processes in the communicator ‘comm’.
     */
    int64_t num_columns;

    /**
     * ‘num_nonzeros’ is the total number of explicitly stored matrix
     * entries for the distributed matrix. This is equal to the sum of
     * the number of explicitly stored matrix entries on each process.
     */
    int64_t num_nonzeros;

    /**
     * ‘rowmapsize’ is the number of matrix rows for which the current
     * process has one or more nonzero entries.
     */
    int rowmapsize;

    /**
     * ‘rowmap’ is an array of length ‘rowmapsize’, containing the
     * global offset of each nonzero matrix row in the part of the
     * matrix owned by the current process. Note that offsets are
     * 0-based, unlike the Matrix Market format, where indices are
     * 1-based.
     *
     * If ‘rowmap’ is ‘NULL’, then every matrix row is nonzero for the
     * part of the matrix owned by the current process. In this case,
     * ‘num_rows’ and ‘rowmapsize’ must be equal, and the matrix rows
     * are implicitly numbered from ‘0’ up to ‘num_rows-1’.
     */
    int64_t * rowmap;

    /**
     * ‘colmapsize’ is the number of matrix columns for which the
     * current process has one or more nonzero entries.
     */
    int colmapsize;

    /**
     * ‘colmap’ is an array of length ‘colmapsize’, containing the
     * global offset of each nonzero matrix column in the part of the
     * matrix owned by the current process. Note that offsets are
     * 0-based, unlike the Matrix Market format, where indices are
     * 1-based.
     *
     * If ‘colmap’ is ‘NULL’, then every matrix column is nonzero for
     * the part of the matrix owned by the current process. In this
     * case, ‘num_columns’ and ‘colmapsize’ must be equal, and the
     * matrix columns are implicitly numbered from ‘0’ up to
     * ‘num_columns-1’.
     */
    int64_t * colmap;

    /**
     * ‘Ap’ is the underlying storage for the part of the matrix
     * belonging to the current process.
     */
    struct mtxmatrix Ap;
};

/**
 * ‘mtxmatrix_dist_field()’ gets the field of a matrix.
 */
int mtxmatrix_dist_field(
    const struct mtxmatrix_dist * A,
    enum mtxfield * field);

/**
 * ‘mtxmatrix_dist_precision()’ gets the precision of a matrix.
 */
int mtxmatrix_dist_precision(
    const struct mtxmatrix_dist * A,
    enum mtxprecision * precision);

/**
 * ‘mtxmatrix_dist_symmetry()’ gets the symmetry of a matrix.
 */
int mtxmatrix_dist_symmetry(
    const struct mtxmatrix_dist * A,
    enum mtxsymmetry * symmetry);

/**
 * ‘mtxmatrix_dist_num_nonzeros()’ gets the number of the number of
 *  nonzero matrix entries, including those represented implicitly due
 *  to symmetry.
 */
int mtxmatrix_dist_num_nonzeros(
    const struct mtxmatrix_dist * A,
    int64_t * num_nonzeros);

/**
 * ‘mtxmatrix_dist_size()’ gets the number of explicitly stored
 * nonzeros of a matrix.
 */
int mtxmatrix_dist_size(
    const struct mtxmatrix_dist * A,
    int64_t * size);

/*
 * Memory management
 */

/**
 * ‘mtxmatrix_dist_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_dist_free(
    struct mtxmatrix_dist * A);

/**
 * ‘mtxmatrix_dist_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_dist_alloc_copy(
    struct mtxmatrix_dist * dst,
    const struct mtxmatrix_dist * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_dist_init_copy(
    struct mtxmatrix_dist * dst,
    const struct mtxmatrix_dist * src,
    struct mtxdisterror * disterr);

/*
 * Initialise matrices from entrywise data in coordinate format with
 * local row and column offsets and explicit mappings from local to
 * global rows and columns.
 */

/**
 * ‘mtxmatrix_dist_alloc_entries_local()’ allocates storage for a
 * matrix based on entrywise data in coordinate format.
 */
int mtxmatrix_dist_alloc_entries_local(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_real_single()’ allocates and
 * initialises a matrix from data in coordinate format with real,
 * single precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_real_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_real_double()’ allocates and
 * initialises a matrix from data in coordinate format with real,
 * double precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_real_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_complex_single()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * single precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_complex_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_complex_double()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * double precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_complex_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_integer_single()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * single precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_integer_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_integer_double()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * double precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_integer_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_local_pattern()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * double precision coefficients.
 */
int mtxmatrix_dist_init_entries_local_pattern(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Initialise matrices from entrywise data in coordinate format with
 * global row and columns offsets.
 */

/**
 * ‘mtxmatrix_dist_alloc_entries_global()’ allocates a distributed
 * matrix, where the local part of the matrix on each process is
 * stored as a matrix of the given type.
 */
int mtxmatrix_dist_alloc_entries_global(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_real_single()’ allocates and
 * initialises a matrix with real, single precision coefficients.
 *
 * On each process, ‘idx’ and ‘data’ are arrays of length
 * ‘num_nonzeros’, containing the global offsets and values,
 * respectively, of the matrix elements stored on the process. Note
 * that ‘num_nonzeros’ may differ from one process to the next.
 */
int mtxmatrix_dist_init_entries_global_real_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_real_double()’ allocates and
 * initialises a matrix with real, double precision coefficients.
 */
int mtxmatrix_dist_init_entries_global_real_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_complex_single()’ allocates and
 * initialises a matrix with complex, single precision coefficients.
 */
int mtxmatrix_dist_init_entries_global_complex_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_complex_double()’ allocates and
 * initialises a matrix with complex, double precision coefficients.
 */
int mtxmatrix_dist_init_entries_global_complex_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_integer_single()’ allocates and
 * initialises a matrix with integer, single precision coefficients.
 */
int mtxmatrix_dist_init_entries_global_integer_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_integer_double()’ allocates and
 * initialises a matrix with integer, double precision coefficients.
 */
int mtxmatrix_dist_init_entries_global_integer_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_pattern()’ allocates and
 * initialises a binary pattern matrix, where every entry has a value
 * of one.
 */
int mtxmatrix_dist_init_entries_global_pattern(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Initialise matrices from strided, entrywise data in coordinate
 * format with global row and columns offsets.
 */

/**
 * ‘mtxmatrix_dist_init_entries_global_strided_real_single()’
 * allocates and initialises a matrix with real, single precision
 * coefficients.
 */
int mtxmatrix_dist_init_entries_global_strided_real_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_strided_real_double()’
 * allocates and initialises a matrix with real, double precision
 * coefficients.
 */
int mtxmatrix_dist_init_entries_global_strided_real_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_strided_complex_single()’
 * allocates and initialises a matrix with complex, single precision
 * coefficients.
 */
int mtxmatrix_dist_init_entries_global_strided_complex_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_strided_complex_double()’
 * allocates and initialises a matrix with complex, double precision
 * coefficients.
 */
int mtxmatrix_dist_init_entries_global_strided_complex_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_strided_integer_single()’
 * allocates and initialises a matrix with integer, single precision
 * coefficients.
 */
int mtxmatrix_dist_init_entries_global_strided_integer_single(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_strided_integer_double()’
 * allocates and initialises a matrix with integer, double precision
 * coefficients.
 */
int mtxmatrix_dist_init_entries_global_strided_integer_double(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_init_entries_global_pattern()’ allocates and
 * initialises a binary pattern matrix, where every entry has a value
 * of one.
 */
int mtxmatrix_dist_init_entries_global_strided_pattern(
    struct mtxmatrix_dist * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Modifying values
 */

/**
 * ‘mtxmatrix_dist_set_constant_real_single()’ sets every nonzero
 * entry of a matrix equal to a constant, single precision floating
 * point number.
 */
int mtxmatrix_dist_set_constant_real_single(
    struct mtxmatrix_dist * A,
    float a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_set_constant_real_double()’ sets every nonzero
 * entry of a matrix equal to a constant, double precision floating
 * point number.
 */
int mtxmatrix_dist_set_constant_real_double(
    struct mtxmatrix_dist * A,
    double a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_set_constant_complex_single()’ sets every nonzero
 * entry of a matrix equal to a constant, single precision floating
 * point complex number.
 */
int mtxmatrix_dist_set_constant_complex_single(
    struct mtxmatrix_dist * A,
    float a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_set_constant_complex_double()’ sets every nonzero
 * entry of a matrix equal to a constant, double precision floating
 * point complex number.
 */
int mtxmatrix_dist_set_constant_complex_double(
    struct mtxmatrix_dist * A,
    double a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_set_constant_integer_single()’ sets every nonzero
 * entry of a matrix equal to a constant integer.
 */
int mtxmatrix_dist_set_constant_integer_single(
    struct mtxmatrix_dist * A,
    int32_t a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_set_constant_integer_double()’ sets every nonzero
 * entry of a matrix equal to a constant integer.
 */
int mtxmatrix_dist_set_constant_integer_double(
    struct mtxmatrix_dist * A,
    int64_t a,
    struct mtxdisterror * disterr);

/*
 * row and column vectors
 */

/**
 * ‘mtxmatrix_dist_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_dist_alloc_row_vector(
    const struct mtxmatrix_dist * A,
    struct mtxmpivector * vector,
    enum mtxvectortype vector_type,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_dist_alloc_column_vector(
    const struct mtxmatrix_dist * A,
    struct mtxmpivector * vector,
    enum mtxvectortype vector_type,
    struct mtxdisterror * disterr);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_dist_from_mtxfile()’ converts from a matrix in Matrix
 * Market format.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxmatrix’ on each process.
 */
int mtxmatrix_dist_from_mtxfile(
    struct mtxmatrix_dist * A,
    const struct mtxfile * mtxfile,
    enum mtxmatrixtype type,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_to_mtxfile()’ converts to a matrix in Matrix Market
 * format.
 */
int mtxmatrix_dist_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_dist * A,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_from_mtxdistfile()’ converts from a matrix in
 * Matrix Market format that is distributed among multiple processes.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxmatrix’ on each process.
 */
int mtxmatrix_dist_from_mtxdistfile(
    struct mtxmatrix_dist * A,
    const struct mtxdistfile * mtxdistfile,
    enum mtxmatrixtype type,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_to_mtxdistfile()’ converts to a matrix in Matrix
 * Market format that is distributed among multiple processes.
 */
int mtxmatrix_dist_to_mtxdistfile(
    struct mtxdistfile * mtxdistfile,
    const struct mtxmatrix_dist * A,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr);

/*
 * I/O operations
 */

/**
 * ‘mtxmatrix_dist_fwrite()’ writes a distributed matrix to a single
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
int mtxmatrix_dist_fwrite(
    const struct mtxmatrix_dist * A,
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
 * ‘mtxmatrix_dist_split()’ splits a matrix into multiple matrices
 * according to a given assignment of parts to each nonzero matrix
 * element.
 *
 * The partitioning of the nonzero matrix elements is specified by the
 * array ‘parts’. The length of the ‘parts’ array is given by ‘size’,
 * which must match the number of explicitly stored nonzero matrix
 * entries in ‘src’. Each entry in the ‘parts’ array is an integer in
 * the range ‘[0, num_parts)’ designating the part to which the
 * corresponding matrix nonzero belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxmatrix_dist’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxmatrix_dist_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxmatrix_dist_split(
    int num_parts,
    struct mtxmatrix_dist ** dsts,
    const struct mtxmatrix_dist * src,
    int64_t size,
    int * parts,
    int64_t * invperm,
    struct mtxdisterror * disterr);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_dist_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both matrices must also have the same number of
 * nonzero elements on that process.
 */
int mtxmatrix_dist_swap(
    struct mtxmatrix_dist * x,
    struct mtxmatrix_dist * y,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both matrices must also have the same number of
 * nonzero elements on that process.
 */
int mtxmatrix_dist_copy(
    struct mtxmatrix_dist * y,
    const struct mtxmatrix_dist * x,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_dist_sscal(
    float a,
    struct mtxmatrix_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_dist_dscal(
    double a,
    struct mtxmatrix_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_dist_cscal(
    float a[2],
    struct mtxmatrix_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmatrix_dist_zscal(
    double a[2],
    struct mtxmatrix_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the dist matrices are
 * allowed.
 */
int mtxmatrix_dist_saxpy(
    float a,
    const struct mtxmatrix_dist * x,
    struct mtxmatrix_dist * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the dist matrices are
 * allowed.
 */
int mtxmatrix_dist_daxpy(
    double a,
    const struct mtxmatrix_dist * x,
    struct mtxmatrix_dist * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the dist matrices are
 * allowed.
 */
int mtxmatrix_dist_saypx(
    float a,
    struct mtxmatrix_dist * y,
    const struct mtxmatrix_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the dist matrices are
 * allowed.
 */
int mtxmatrix_dist_daypx(
    double a,
    struct mtxmatrix_dist * y,
    const struct mtxmatrix_dist * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_sdot()’ computes the Euclidean dot product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_sdot(
    const struct mtxmatrix_dist * x,
    const struct mtxmatrix_dist * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_ddot()’ computes the Euclidean dot product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_ddot(
    const struct mtxmatrix_dist * x,
    const struct mtxmatrix_dist * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_cdotu(
    const struct mtxmatrix_dist * x,
    const struct mtxmatrix_dist * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_zdotu(
    const struct mtxmatrix_dist * x,
    const struct mtxmatrix_dist * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_cdotc()’ computes the Euclidean dot product of two
 * complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_cdotc(
    const struct mtxmatrix_dist * x,
    const struct mtxmatrix_dist * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_zdotc()’ computes the Euclidean dot product of two
 * complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_zdotc(
    const struct mtxmatrix_dist * x,
    const struct mtxmatrix_dist * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_snrm2()’ computes the Euclidean norm of a matrix in
 * single precision floating point. Repeated indices in the dist
 * matrix are not allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_snrm2(
    const struct mtxmatrix_dist * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_dnrm2()’ computes the Euclidean norm of a matrix in
 * double precision floating point. Repeated indices in the dist
 * matrix are not allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_dnrm2(
    const struct mtxmatrix_dist * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist matrix are not allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_sasum(
    const struct mtxmatrix_dist * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist matrix are not allowed, otherwise the result is undefined.
 */
int mtxmatrix_dist_dasum(
    const struct mtxmatrix_dist * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts. Repeated indices in the dist matrix are not allowed,
 * otherwise the result is undefined.
 */
int mtxmatrix_dist_iamax(
    const struct mtxmatrix_dist * x,
    int * iamax,
    struct mtxdisterror * disterr);

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxmatrix_dist_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_dist_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_dist * A,
    const struct mtxmpivector * x,
    float beta,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_dgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_dist_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_dist * A,
    const struct mtxmpivector * x,
    double beta,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmatrix_dist_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_dist * A,
    const struct mtxmpivector * x,
    float beta[2],
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmatrix_dist_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_dist * A,
    const struct mtxmpivector * x,
    double beta[2],
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/*
 * persistent matrix-vector multiply operations, with optional
 * overlapping of computation and communication
 */

struct mtxmatrix_dist_gemv_impl;

/**
 * ‘mtxmatrix_dist_gemv’ is a data structure for a persistent,
 * matrix-vector multiply operation.
 */
struct mtxmatrix_dist_gemv
{
    enum mtxtransposition trans;
    const struct mtxmatrix_dist * A;
    const struct mtxmpivector * x;
    struct mtxmpivector * y;
    enum mtxgemvoverlap overlap;
    struct mtxmatrix_dist_gemv_impl * impl;
};

/**
 * ‘mtxmatrix_dist_gemv_init()’ allocates data structures for a
 * persistent, matrix-vector multiply operation.
 *
 * This is used in cases where the matrix-vector multiply operation is
 * performed repeatedly, since the setup phase only needs to be
 * carried out once.
 */
int mtxmatrix_dist_gemv_init(
    struct mtxmatrix_dist_gemv * gemv,
    enum mtxtransposition trans,
    const struct mtxmatrix_dist * A,
    const struct mtxmpivector * x,
    struct mtxmpivector * y,
    enum mtxgemvoverlap overlap,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_gemv_free()’ frees resources associated with a
 * persistent, matrix-vector multiply operation.
 */
void mtxmatrix_dist_gemv_free(
    struct mtxmatrix_dist_gemv * gemv);

/**
 * ‘mtxmatrix_dist_gemv_sgemv()’ initiates a matrix-vector multiply
 * operation to multiply a matrix ‘A’ or its transpose ‘A'’ by a real
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another real scalar ‘beta’
 * (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmatrix_dist_gemv_wait()’
 * is called.
 */
int mtxmatrix_dist_gemv_sgemv(
    struct mtxmatrix_dist_gemv * gemv,
    float alpha,
    float beta,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_gemv_dgemv()’ initiates a matrix-vector multiply
 * operation to multiply a matrix ‘A’ or its transpose ‘A'’ by a real
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another real scalar ‘beta’
 * (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmatrix_dist_gemv_wait()’
 * is called.
 */
int mtxmatrix_dist_gemv_dgemv(
    struct mtxmatrix_dist_gemv * gemv,
    double alpha,
    double beta,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_gemv_cgemv()’ initiates a matrix-vector multiply
 * operation to multiply a complex-values matrix ‘A’, its transpose
 * ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar ‘alpha’
 * (‘α’) and a vector ‘x’, before adding the result to another vector
 * ‘y’ multiplied by another complex ‘beta’ (‘β’). That is, ‘y = α*A*x
 * + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmatrix_dist_gemv_wait()’
 * is called.
 */
int mtxmatrix_dist_gemv_cgemv(
    struct mtxmatrix_dist_gemv * gemv,
    float alpha[2],
    float beta[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_gemv_zgemv()’ initiates a matrix-vector multiply
 * operation to multiply a complex-values matrix ‘A’, its transpose
 * ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar ‘alpha’
 * (‘α’) and a vector ‘x’, before adding the result to another vector
 * ‘y’ multiplied by another complex ‘beta’ (‘β’). That is, ‘y = α*A*x
 * + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmatrix_dist_gemv_wait()’
 * is called.
 */
int mtxmatrix_dist_gemv_zgemv(
    struct mtxmatrix_dist_gemv * gemv,
    double alpha[2],
    double beta[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmatrix_dist_gemv_wait()’ waits for a persistent, matrix-vector
 * multiply operation to finish.
 */
int mtxmatrix_dist_gemv_wait(
    struct mtxmatrix_dist_gemv * gemv,
    int64_t * num_flops,
    struct mtxdisterror * disterr);
#endif
#endif
