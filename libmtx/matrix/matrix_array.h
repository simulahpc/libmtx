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
 * Last modified: 2022-03-15
 *
 * Data structures for matrices in array format.
 */

#ifndef LIBMTX_MATRIX_ARRAY_H
#define LIBMTX_MATRIX_ARRAY_H

#include <libmtx/libmtx-config.h>

#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/util/symmetry.h>
#include <libmtx/util/transpose.h>
#include <libmtx/vector/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxmatrix;
struct mtxpartition;
struct mtxvector;

/**
 * ‘mtxmatrix_array’ represents a matrix in array format.
 */
struct mtxmatrix_array
{
    /**
     * ‘field’ is the matrix field: ‘real’, ‘complex’, ‘integer’ or
     * ‘pattern’.
     */
    enum mtxfield field;

    /**
     * ‘precision’ is the precision used to store values.
     */
    enum mtxprecision precision;

    /**
     * ‘symmetry’ is the matrix symmetry: ‘unsymmetric’, ‘symmetric’,
     * ‘skew-symmetric’ or ‘hermitian’.
     */
    enum mtxsymmetry symmetry;

    /**
     * ‘num_rows’ is the number of matrix rows.
     */
    int num_rows;

    /**
     * ‘num_columns’ is the number of matrix columns.
     */
    int num_columns;

    /**
     * ‘size’ is the number of matrix elements, which is equal to
     * ‘num_rows*num_columns’.
     */
    int64_t size;

    /**
     * ‘data’ contains values for each matrix entry.
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
 * ‘mtxmatrix_array_alloc()’ allocates a matrix in array format.
 */
int mtxmatrix_array_alloc(
    struct mtxmatrix_array * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns);

/**
 * ‘mtxmatrix_array_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_array_free(
    struct mtxmatrix_array * matrix);

/**
 * ‘mtxmatrix_array_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_array_alloc_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/**
 * ‘mtxmatrix_array_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_array_init_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/*
 * Matrix initialisation
 */

/**
 * ‘mtxmatrix_array_init_real_single()’ allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_array_init_real_single(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data);

/**
 * ‘mtxmatrix_array_init_real_double()’ allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_array_init_real_double(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data);

/**
 * ‘mtxmatrix_array_init_complex_single()’ allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_array_init_complex_single(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2]);

/**
 * ‘mtxmatrix_array_init_complex_double()’ allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_array_init_complex_double(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2]);

/**
 * ‘mtxmatrix_array_init_integer_single()’ allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_array_init_integer_single(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data);

/**
 * ‘mtxmatrix_array_init_integer_double()’ allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_array_init_integer_double(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data);

/*
 * Row and column vectors
 */

/**
 * ‘mtxmatrix_array_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_array_alloc_row_vector(
    const struct mtxmatrix_array * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);

/**
 * ‘mtxmatrix_array_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_array_alloc_column_vector(
    const struct mtxmatrix_array * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_array_from_mtxfile()’ converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_array_from_mtxfile(
    struct mtxmatrix_array * matrix,
    const struct mtxfile * mtxfile);

/**
 * ‘mtxmatrix_array_to_mtxfile()’ converts a matrix to a matrix in
 * Matrix Market format.
 */
int mtxmatrix_array_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_array * matrix,
    enum mtxfileformat mtxfmt);

/*
 * Nonzero rows and columns
 */

/**
 * ‘mtxmatrix_array_nzrows()’ counts the number of nonzero (non-empty)
 * matrix rows, and, optionally, fills an array with the row indices
 * of the nonzero (non-empty) matrix rows.
 *
 * If ‘num_nonzero_rows’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix rows.
 *
 * ‘nonzero_rows’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the row
 * indices of the nonzero matrix rows. Note that ‘size’ must be at
 * least equal to the number of non-zero rows.
 */
int mtxmatrix_array_nzrows(
    const struct mtxmatrix_array * matrix,
    int * num_nonzero_rows,
    int size,
    int * nonzero_rows);

/**
 * ‘mtxmatrix_array_nzcols()’ counts the number of nonzero (non-empty)
 * matrix columns, and, optionally, fills an array with the column
 * indices of the nonzero (non-empty) matrix columns.
 *
 * If ‘num_nonzero_columns’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix columns.
 *
 * ‘nonzero_columns’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the column
 * indices of the nonzero matrix columns. Note that ‘size’ must be at
 * least equal to the number of non-zero columns.
 */
int mtxmatrix_array_nzcols(
    const struct mtxmatrix_array * matrix,
    int * num_nonzero_columns,
    int size,
    int * nonzero_columns);

/*
 * Partitioning
 */

/**
 * ‘mtxmatrix_array_partition()’ partitions a matrix into blocks
 * according to the given row and column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix ‘src’, respectively. That is, ‘rowpart->size’
 * must be equal to the number of matrix rows, and ‘colpart->size’
 * must be equal to the number of matrix columns.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxmatrix’, where ‘P’ is the number of
 * row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * matrix in the ‘dsts’ array.
 */
int mtxmatrix_array_partition(
    struct mtxmatrix * dsts,
    const struct mtxmatrix_array * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart);

/**
 * ‘mtxmatrix_array_join()’ joins together matrices representing
 * compatible blocks of a partitioned matrix to form a larger matrix.
 *
 * The argument ‘srcs’ is logically arranged as a two-dimensional
 * array of size ‘P*Q’, where ‘P’ is the number of row parts
 * (‘rowpart->num_parts’) and ‘Q’ is the number of column parts
 * (‘colpart->num_parts’).  Note that the ‘r’th part corresponds to a
 * row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus, the
 * ‘r’th entry of ‘srcs’ is the submatrix corresponding to the ‘p’th
 * row and ‘q’th column of the 2D partitioning.
 *
 * Moreover, the blocks must be compatible, which means that each part
 * in the same block row ‘p’, must have the same number of rows.
 * Similarly, each part in the same block column ‘q’ must have the
 * same number of columns. Finally, for each block column ‘q’, the sum
 * of the number of rows of ‘srcs[p*Q+q]’ for ‘p=0,1,...,P-1’ must be
 * equal to ‘rowpart->size’. Likewise, for each block row ‘p’, the sum
 * of the number of columns of ‘srcs[p*Q+q]’ for ‘q=0,1,...,Q-1’ must
 * be equal to ‘colpart->size’.
 */
int mtxmatrix_array_join(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix * srcs,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmatrix_array_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_swap(
    struct mtxmatrix_array * x,
    struct mtxmatrix_array * y);

/**
 * ‘mtxmatrix_array_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_copy(
    struct mtxmatrix_array * y,
    const struct mtxmatrix_array * x);

/**
 * ‘mtxmatrix_array_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_array_sscal(
    float a,
    struct mtxmatrix_array * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_array_dscal(
    double a,
    struct mtxmatrix_array * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_saxpy(
    float a,
    const struct mtxmatrix_array * x,
    struct mtxmatrix_array * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_daxpy(
    double a,
    const struct mtxmatrix_array * x,
    struct mtxmatrix_array * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_saypx(
    float a,
    struct mtxmatrix_array * y,
    const struct mtxmatrix_array * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_daypx(
    double a,
    struct mtxmatrix_array * y,
    const struct mtxmatrix_array * x,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_sdot(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    float * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_ddot(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_cdotu(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_zdotu(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_cdotc(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    float (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_zdotc(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    double (* dot)[2],
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_array_snrm2(
    const struct mtxmatrix_array * x,
    float * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_array_dnrm2(
    const struct mtxmatrix_array * x,
    double * nrm2,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_array_sasum(
    const struct mtxmatrix_array * x,
    float * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_array_dasum(
    const struct mtxmatrix_array * x,
    double * asum,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_array_iamax(
    const struct mtxmatrix_array * x,
    int * iamax);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_array_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_dgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

/**
 * ‘mtxmatrix_array_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops);

#endif
