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
 * Data structures for matrices in array format.
 */

#ifndef LIBMTX_MATRIX_ARRAY_H
#define LIBMTX_MATRIX_ARRAY_H

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
 * `mtxmatrix_array' represents a matrix in array format.
 */
struct mtxmatrix_array
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
     * `size' is the number of matrix elements, which is equal to
     * ‘num_rows*num_columns’.
     */
    int64_t size;

    /**
     * `data' contains values for each matrix entry.
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
 * `mtxmatrix_array_alloc()' allocates a matrix in array format.
 */
int mtxmatrix_array_alloc(
    struct mtxmatrix_array * matrix,
    enum mtx_field_ field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns);

/**
 * `mtxmatrix_array_free()' frees storage allocated for a matrix.
 */
void mtxmatrix_array_free(
    struct mtxmatrix_array * matrix);

/**
 * `mtxmatrix_array_alloc_copy()' allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_array_alloc_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/**
 * `mtxmatrix_array_init_copy()' allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_array_init_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/*
 * Matrix initialisation
 */

/**
 * `mtxmatrix_array_init_real_single()' allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_array_init_real_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float * data);

/**
 * `mtxmatrix_array_init_real_double()' allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_array_init_real_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double * data);

/**
 * `mtxmatrix_array_init_complex_single()' allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_array_init_complex_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float (* data)[2]);

/**
 * `mtxmatrix_array_init_complex_double()' allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_array_init_complex_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double (* data)[2]);

/**
 * `mtxmatrix_array_init_integer_single()' allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_array_init_integer_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int32_t * data);

/**
 * `mtxmatrix_array_init_integer_double()' allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_array_init_integer_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int64_t * data);

/*
 * Row and column vectors
 */

/**
 * `mtxmatrix_array_alloc_row_vector()' allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_array_alloc_row_vector(
    const struct mtxmatrix_array * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type);

/**
 * `mtxmatrix_array_alloc_column_vector()' allocates a column vector
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
 * `mtxmatrix_array_from_mtxfile()' converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_array_from_mtxfile(
    struct mtxmatrix_array * matrix,
    const struct mtxfile * mtxfile);

/**
 * `mtxmatrix_array_to_mtxfile()' converts a matrix to a matrix in
 * Matrix Market format.
 */
int mtxmatrix_array_to_mtxfile(
    const struct mtxmatrix_array * matrix,
    struct mtxfile * mtxfile);

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
 * the size of ‘x’ must equal the number of rows of ‘A’ and the
 * size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y);

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
 * the size of ‘x’ must equal the number of rows of ‘A’ and the
 * size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y);

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
 * the size of ‘x’ must equal the number of rows of ‘A’ and the
 * size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y);

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
 * the size of ‘x’ must equal the number of rows of ‘A’ and the
 * size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y);

#endif
