/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-16
 *
 * Data structures for matrices in coordinate format.
 */

#ifndef LIBMTX_MATRIX_COORDINATE_DATA_H
#define LIBMTX_MATRIX_COORDINATE_DATA_H

#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/triangle.h>

#include <stdint.h>

struct mtx_index_set;

/*
 * Data types for coordinate matrix values.
 */

/**
 * `mtx_matrix_coordinate_real_single' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `real' field, when using single precision data types.
 */
struct mtx_matrix_coordinate_real_single
{
    int i;    /* row index */
    int j;    /* column index */
    float a;  /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_double' represents a nonzero matrix entry in
 * a Matrix Market file with `matrix' object, `coordinate' format and
 * `real' field, when using double precision data types.
 */
struct mtx_matrix_coordinate_real_double
{
    int i;     /* row index */
    int j;     /* column index */
    double a;  /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_complex_single' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `complex' field, when using single precision data types.
 */
struct mtx_matrix_coordinate_complex_single
{
    int i;       /* row index */
    int j;       /* column index */
    float a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtx_matrix_coordinate_complex_double' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `complex' field, when using double precision data types.
 */
struct mtx_matrix_coordinate_complex_double
{
    int i;       /* row index */
    int j;       /* column index */
    double a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtx_matrix_coordinate_integer_single' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `integer' field, when using single precision data types.
 */
struct mtx_matrix_coordinate_integer_single
{
    int i;      /* row index */
    int j;      /* column index */
    int32_t a;  /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_integer_double' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `integer' field, when using double precision data types.
 */
struct mtx_matrix_coordinate_integer_double
{
    int i;      /* row index */
    int j;      /* column index */
    int64_t a;  /* nonzero value */
};

/**
 * `mtx_matrix_coordinate_pattern' represents a nonzero matrix entry
 * in a Matrix Market file with `matrix' object, `coordinate' format
 * and `pattern' field.
 */
struct mtx_matrix_coordinate_pattern
{
    int i;  /* row index */
    int j;  /* column index */
};

/**
 * `mtx_matrix_coordinate_data' is a data structure for representing
 * data associated with matrices in coordinate format.
 */
struct mtx_matrix_coordinate_data
{
    /**
     * `field' is the field associated with the matrix values: `real',
     * `complex', `integer' or `pattern'.
     */
    enum mtx_field field;

    /**
     * `precision' is the precision associated with the matrix values:
     * `single' or `double'.
     */
    enum mtx_precision precision;

    /**
     * `symmetry' is the matrix symmetry: `general', `symmetric',
     * `skew-symmetric', or `hermitian'.
     *
     * Note that if `symmetry' is `symmetric', `skew-symmetric' or
     * `hermitian', then the matrix must be square, so that `num_rows'
     * is equal to `num_columns'.
     */
    enum mtx_symmetry symmetry;

    /**
     * `triangle' specifies triangular properties of a matrix:
     * `mtx_nontriangular', `mtx_lower_triangular',
     * `mtx_upper_triangular', `mtx_strict_lower_triangular' or
     * `mtx_strict_upper_triangular'.
     *
     * `triangle' is `mtx_lower_triangular' if only the lower
     * triangular part of the matrix is stored, or
     * `mtx_upper_triangular' if only the upper triangular part of the
     * matrix is stored.  However, if `triangle' is
     * `mtx_nontriangular', then the matrix nonzeros may belong to
     * either the upper or lower triangular part of the matrix.
     *
     * For skew-symmetric matrices, `triangle' is
     * `mtx_strict_lower_triangular' if only the strict lower
     * triangular part of the matrix is stored, or
     * `mtx_strict_upper_triangular' if only the strict upper
     * triangular part of the matrix is stored.  However, if
     * `triangle' is `mtx_nontriangular', then the matrix nonzeros may
     * belong to either the strict upper or lower triangular part of
     * the matrix.  If there are any entries on the diagonal, they
     * must be zero.
     *
     * Note that the triangular properties of a matrix are not
     * explicitly stored in a Matrix Market file, but it is useful
     * additional data that can be provided by the user.
     */
    enum mtx_triangle triangle;

    /**
     * `sorting' is the sorting of matrix nonzeros: `unsorted',
     * 'row-major' or 'column-major'.
     *
     * Note that the sorting is not explicitly stored in a Matrix
     * Market file, but it is useful additional data that can be
     * provided by the user.
     */
    enum mtx_sorting sorting;

    /**
     * `assembly' is the matrix assembly state: `unassembled' or
     * `assembled'.
     *
     * An unassembled sparse matrix may contain more than one value
     * associated with each nonzero matrix entry. In contrast, there
     * is only one value associated with each nonzero matrix entry of
     * an assembled sparse matrix.
     *
     * Note that the assembly state is not explicitly stored in a
     * Matrix Market file, but it is useful additional data that can
     * be provided by the user.
     */
    enum mtx_assembly assembly;

    /**
     * `num_rows' is the number of rows in the matrix.
     */
    int num_rows;

    /**
     * `num_columns' is the number of columns in the matrix.
     */
    int num_columns;

    /**
     * `size' is the number of entries stored in the `data' array.
     */
    int64_t size;

    /**
     * `data' is used to store the matrix values.
     *
     * The storage format of nonzero values depends on `field' and
     * `precision'.  Only the member of the `data' union that
     * corresponds to the matrix `field' and `precision' should be
     * used to access the data.
     *
     * For example, if `field' is `real' and `precision' is `single',
     * then `data.real_single' is an array of `size' values of type
     * `struct mtx_matrix_coordinate_real_single', which contains the
     * locations and values of the matrix entries.
     */
    union {
        struct mtx_matrix_coordinate_real_single * real_single;
        struct mtx_matrix_coordinate_real_double * real_double;
        struct mtx_matrix_coordinate_complex_single * complex_single;
        struct mtx_matrix_coordinate_complex_double * complex_double;
        struct mtx_matrix_coordinate_integer_single * integer_single;
        struct mtx_matrix_coordinate_integer_double * integer_double;
        struct mtx_matrix_coordinate_pattern * pattern;
    } data;
};

/**
 * `mtx_matrix_coordinate_data_free()' frees resources associated with
 * the matrix data in coordinate format.
 */
void mtx_matrix_coordinate_data_free(
    struct mtx_matrix_coordinate_data * mtxdata);

/**
 * `mtx_matrix_coordinate_data_alloc()' allocates data for a matrix in
 * coordinate format.
 */
int mtx_matrix_coordinate_data_alloc(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t size);

/*
 * Coordinate matrix allocation and initialisation.
 */

/**
 * `mtx_matrix_coordinate_data_init_real_single()' creates data for a
 * matrix with real, single-precision floating point coefficients.
 */
int mtx_matrix_coordinate_data_init_real_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_single * data);

/**
 * `mtx_matrix_coordinate_data_init_real_double()' creates data for a
 * matrix with real, double-precision floating point coefficients.
 */
int mtx_matrix_coordinate_data_init_real_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_real_double * data);

/**
 * `mtx_matrix_coordinate_data_init_complex_single()' creates data for
 * a matrix with complex, single-precision floating point
 * coefficients.
 */
int mtx_matrix_coordinate_data_init_complex_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex_single * data);

/**
 * `mtx_matrix_coordinate_data_init_complex_double()' creates data for
 * a matrix with complex, double-precision floating point
 * coefficients.
 */
int mtx_matrix_coordinate_data_init_complex_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_complex_double * data);

/**
 * `mtx_matrix_coordinate_data_init_integer_single()' creates data for
 * a matrix with integer, single-precision coefficients.
 */
int mtx_matrix_coordinate_data_init_integer_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer_single * data);

/**
 * `mtx_matrix_coordinate_data_init_integer_double()' creates data for
 * a matrix with integer, double-precision coefficients.
 */
int mtx_matrix_coordinate_data_init_integer_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_integer_double * data);

/**
 * `mtx_matrix_coordinate_data_init_pattern()' creates data for a
 * matrix with boolean (pattern) coefficients.
 */
int mtx_matrix_coordinate_data_init_pattern(
    struct mtx_matrix_coordinate_data * mtxdata,
    enum mtx_symmetry symmetry,
    enum mtx_triangle triangle,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_matrix_coordinate_pattern * data);

/**
 * `mtx_matrix_coordinate_data_copy_alloc()' allocates a copy of a
 * matrix without copying the matrix values.
 */
int mtx_matrix_coordinate_data_copy_alloc(
    struct mtx_matrix_coordinate_data * dst,
    const struct mtx_matrix_coordinate_data * src);

/**
 * `mtx_matrix_coordinate_data_copy_init()' creates a copy of a matrix
 * and also copies matrix values.
 */
int mtx_matrix_coordinate_data_copy_init(
    struct mtx_matrix_coordinate_data * dst,
    const struct mtx_matrix_coordinate_data * src);

/**
 * `mtx_matrix_coordinate_data_set_zero()' zeroes a matrix.
 */
int mtx_matrix_coordinate_data_set_zero(
    struct mtx_matrix_coordinate_data * mtxdata);

/**
 * `mtx_matrix_coordinate_data_set_constant_real_single()' sets every
 * (nonzero) value of a matrix equal to a constant, single precision
 * floating point number.
 */
int mtx_matrix_coordinate_data_set_constant_real_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    float a);

/**
 * `mtx_matrix_coordinate_data_set_constant_real_double()' sets every
 * (nonzero) value of a matrix equal to a constant, double precision
 * floating point number.
 */
int mtx_matrix_coordinate_data_set_constant_real_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    double a);

/**
 * `mtx_matrix_coordinate_data_set_constant_complex_single()' sets
 * every (nonzero) value of a matrix equal to a constant, single
 * precision floating point complex number.
 */
int mtx_matrix_coordinate_data_set_constant_complex_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    float a[2]);

/**
 * `mtx_matrix_coordinate_data_set_constant_complex_double()' sets
 * every (nonzero) value of a matrix equal to a constant, double
 * precision floating point complex number.
 */
int mtx_matrix_coordinate_data_set_constant_complex_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    double a[2]);

/**
 * `mtx_matrix_coordinate_data_set_constant_integer_single()' sets
 * every (nonzero) value of a matrix equal to a constant integer.
 */
int mtx_matrix_coordinate_data_set_constant_integer_single(
    struct mtx_matrix_coordinate_data * mtxdata,
    int32_t a);

/**
 * `mtx_matrix_coordinate_data_set_constant_integer_double()' sets
 * every (nonzero) value of a matrix equal to a constant integer.
 */
int mtx_matrix_coordinate_data_set_constant_integer_double(
    struct mtx_matrix_coordinate_data * mtxdata,
    int64_t a);

/*
 * Other functions.
 */

/**
 * `mtx_matrix_coordinate_data_size_per_row()' counts the number of
 * entries stored for each row of a matrix.
 *
 * The array `size_per_row' must point to an array containing enough
 * storage for `num_rows' values of type `int'.
 */
int mtx_matrix_coordinate_data_size_per_row(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int num_rows,
    int32_t * size_per_row);

/**
 * `mtx_matrix_coordinate_data_diagonals_per_row()' counts for each
 * row of a matrix the number of nonzero entries on the diagonal that
 * are stored.  If the matrix is not in an assembled state, then the
 * count will also count any duplicate diagonal entries.
 *
 * The array `diagonals_per_row' must point to an array containing
 * enough storage for `mtxdata->num_rows' values of type `int'.
 */
int mtx_matrix_coordinate_data_diagonals_per_row(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int num_rows,
    int * diagonals_per_row);

/**
 * `mtx_matrix_coordinate_data_column_ptr()' computes column pointers
 * of a matrix.
 *
 * The array `column_ptr' must point to an array containing enough
 * storage for `mtxdata->num_columns+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in column major order.  If
 * the matrix is sorted in column major order, then the `i'-th entry
 * of `column_ptr' is the location of the first nonzero in the
 * `mtxdata->data' array that belongs to the `i+1'-th column of the
 * matrix, for `i=0,1,...,num_columns-1'.  The final entry of
 * `column_ptr' indicates the position one place beyond the last entry
 * in `mtxdata->data'.
 */
int mtx_matrix_coordinate_data_column_ptr(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int size,
    int64_t * column_ptr);

/**
 * `mtx_matrix_coordinate_data_row_ptr()' computes row pointers of a
 * matrix.
 *
 * The array `row_ptr' must point to an array containing enough
 * storage for `mtxdata->num_rows+1' values of type `int64_t'.
 *
 * The matrix is not required to be sorted in row major order.  If the
 * matrix is sorted in row major order, then the `i'-th entry of
 * `row_ptr' is the location of the first nonzero in the
 * `mtxdata->data' array that belongs to the `i+1'-th row of the
 * matrix, for `i=0,1,...,num_rows-1'. The final entry of `row_ptr'
 * indicates the position one place beyond the last entry in
 * `mtxdata->data'.
 */
int mtx_matrix_coordinate_data_row_ptr(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int size,
    int64_t * row_ptr);

/**
 * `mtx_matrix_coordinate_data_column_indices()' extracts the column
 * indices of a matrix to a separate array.
 *
 * The array `column_indices' must point to an array containing enough
 * storage for `size' values of type `int'.
 */
int mtx_matrix_coordinate_data_column_indices(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int64_t size,
    int * column_indices);

/**
 * `mtx_matrix_coordinate_data_row_indices()' extracts the row indices
 * of a matrix to a separate array.
 *
 * The array `row_indices' must point to an array containing enough
 * storage for `size' values of type `int'.
 */
int mtx_matrix_coordinate_data_row_indices(
    const struct mtx_matrix_coordinate_data * mtxdata,
    int64_t size,
    int * row_indices);

/**
 * `mtx_matrix_coordinate_data_submatrix()' obtains a submatrix
 * consisting of the given rows and columns.
 */
int mtx_matrix_coordinate_data_submatrix(
    struct mtx_matrix_coordinate_data * submtx,
    const struct mtx_matrix_coordinate_data * mtx,
    const struct mtx_index_set * rows,
    const struct mtx_index_set * columns);

/**
 * `mtx_matrix_coordinate_data_transpose()' transposes a coordinate
 * matrix.
 */
int mtx_matrix_coordinate_data_transpose(
    struct mtx_matrix_coordinate_data * mtxdata);

#endif
