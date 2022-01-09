/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-10-05
 *
 * Data structures for matrices in coordinate format.
 */

#ifndef LIBMTX_MATRIX_COORDINATE_H
#define LIBMTX_MATRIX_COORDINATE_H

#include <libmtx/libmtx-config.h>

#include <libmtx/mtx/precision.h>
#include <libmtx/util/field.h>
#include <libmtx/util/transpose.h>
#include <libmtx/vector/vector.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct mtxfile;
struct mtxvector;

/**
 * `mtxmatrix_coordinate' represents a matrix in coordinate format.
 */
struct mtxmatrix_coordinate
{
    /**
     * `field' is the matrix field: `real', `complex', `integer' or
     * `pattern'.
     */
    enum mtx_field_ field;

    /**
     * `precision' is the precision used to store values.
     */
    enum mtxprecision precision;

    /**
     * `num_rows' is the number of matrix rows.
     */
    int num_rows;

    /**
     * `num_columns' is the number of matrix columns.
     */
    int num_columns;

    /**
     * `num_nonzeros' is the number of nonzero matrix entries for a
     * sparse matrix.
     */
    int64_t num_nonzeros;

    /**
     * `rowidx' is an array containing the row indices of nonzero
     * matrix entries.  Note that row indices are 0-based, unlike the
     * Matrix Market format, where indices are 1-based.
     */
    int * rowidx;

    /**
     * `colidx' is an array containing the column indices of nonzero
     * matrix entries.  Note that column indices are 0-based, unlike
     * the Matrix Market format, where indices are 1-based.
     */
    int * colidx;

    /**
     * `data' contains the values of nonzero matrix entries.
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
 * `mtxmatrix_coordinate_free()' frees storage allocated for a matrix.
 */
void mtxmatrix_coordinate_free(
    struct mtxmatrix_coordinate * matrix);

/**
 * `mtxmatrix_coordinate_alloc_copy()' allocates a copy of a matrix
 * without initialising the values.
 */
int mtxmatrix_coordinate_alloc_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src);

/**
 * `mtxmatrix_coordinate_init_copy()' allocates a copy of a matrix and
 * also copies the values.
 */
int mtxmatrix_coordinate_init_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src);

/*
 * Matrix coordinate formats
 */

/**
 * `mtxmatrix_coordinate_alloc()' allocates a matrix in coordinate
 * format.
 */
int mtxmatrix_coordinate_alloc(
    struct mtxmatrix_coordinate * matrix,
    enum mtx_field_ field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros);

/**
 * `mtxmatrix_coordinate_init_real_single()' allocates and initialises
 * a matrix in coordinate format with real, single precision
 * coefficients.
 */
int mtxmatrix_coordinate_init_real_single(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data);

/**
 * `mtxmatrix_coordinate_init_real_double()' allocates and initialises
 * a matrix in coordinate format with real, double precision
 * coefficients.
 */
int mtxmatrix_coordinate_init_real_double(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data);

/**
 * `mtxmatrix_coordinate_init_complex_single()' allocates and
 * initialises a matrix in coordinate format with complex, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_complex_single(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2]);

/**
 * `mtxmatrix_coordinate_init_complex_double()' allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_complex_double(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2]);

/**
 * `mtxmatrix_coordinate_init_integer_single()' allocates and
 * initialises a matrix in coordinate format with integer, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_integer_single(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data);

/**
 * `mtxmatrix_coordinate_init_integer_double()' allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_integer_double(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data);

/**
 * `mtxmatrix_coordinate_init_pattern()' allocates and initialises a
 * matrix in coordinate format with boolean coefficients.
 */
int mtxmatrix_coordinate_init_pattern(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx);

/*
 * Row and column vectors
 */

/**
 * `mtxmatrix_coordinate_alloc_row_vector()' allocates a row vector
 * for a given matrix, where a row vector is a vector whose length
 * equal to a single row of the matrix.
 */
int mtxmatrix_coordinate_alloc_row_vector(
    const struct mtxmatrix_coordinate * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);

/**
 * `mtxmatrix_coordinate_alloc_column_vector()' allocates a column
 * vector for a given matrix, where a column vector is a vector whose
 * length equal to a single column of the matrix.
 */
int mtxmatrix_coordinate_alloc_column_vector(
    const struct mtxmatrix_coordinate * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxmatrix_coordinate_from_mtxfile()' converts a matrix in Matrix
 * Market format to a matrix.
 */
int mtxmatrix_coordinate_from_mtxfile(
    struct mtxmatrix_coordinate * matrix,
    const struct mtxfile * mtxfile);

/**
 * `mtxmatrix_coordinate_to_mtxfile()' converts a matrix to a matrix
 * in Matrix Market format.
 */
int mtxmatrix_coordinate_to_mtxfile(
    const struct mtxmatrix_coordinate * matrix,
    struct mtxfile * mtxfile);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_coordinate_sgemv()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision.  If the field of the matrix ‘A’
 * is ‘real’, ‘integer’ or ‘complex’, then the vectors must have the
 * same field.  Otherwise, if the matrix field is ‘pattern’, then ‘x’
 * and ‘y’ are allowed to be ‘real’, ‘integer’ or ‘complex’, but they
 * must both have the same field.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y);

/**
 * ‘mtxmatrix_coordinate_dgemv()’ multiplies a matrix ‘A’ or its
 * transpose ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’,
 * before adding the result to another vector ‘y’ multiplied by
 * another scalar real ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y
 * = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision.  If the field of the matrix ‘A’
 * is ‘real’, ‘integer’ or ‘complex’, then the vectors must have the
 * same field.  Otherwise, if the matrix field is ‘pattern’, then ‘x’
 * and ‘y’ are allowed to be ‘real’, ‘integer’ or ‘complex’, but they
 * must both have the same field.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y);

/**
 * ‘mtxmatrix_coordinate_cgemv()’ multiplies a complex-valued matrix
 * ‘A’, its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a
 * complex scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another complex scalar
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y
 * = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision as the matrix ‘A’.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y);

/**
 * ‘mtxmatrix_coordinate_zgemv()’ multiplies a complex-valued matrix
 * ‘A’, its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a
 * complex scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the
 * result to another vector ‘y’ multiplied by another complex scalar
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y
 * = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must both
 * have the same field and precision as the matrix ‘A’.
 *
 * Moreover, if ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must
 * equal the number of columns of ‘A’ and the size of ‘y’ must equal
 * the number of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or
 * ‘mtx_conjtrans’, then the size of ‘x’ must equal the number of rows
 * of ‘A’ and the size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y);

#endif
