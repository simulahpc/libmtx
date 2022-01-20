/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-16
 *
 * Data structures for vectors in coordinate format.
 */

#ifndef LIBMTX_VECTOR_COORDINATE_DATA_H
#define LIBMTX_VECTOR_COORDINATE_DATA_H

#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/precision.h>
#include <libmtx/mtx/sort.h>

#include <stdint.h>

/*
 * Data types for coordinate vector values.
 */

/**
 * `mtx_vector_coordinate_real_single' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `real' field, when using single precision data types.
 */
struct mtx_vector_coordinate_real_single
{
    int i;    /* row index */
    float a;  /* nonzero value */
};

/**
 * `mtx_vector_coordinate_double' represents a nonzero vector entry in
 * a Matrix Market file with `vector' object, `coordinate' format and
 * `real' field, when using double precision data types.
 */
struct mtx_vector_coordinate_real_double
{
    int i;    /* row index */
    double a; /* nonzero value */
};

/**
 * `mtx_vector_coordinate_complex_single' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `complex' field, when using single precision data types.
 */
struct mtx_vector_coordinate_complex_single
{
    int i;        /* row index */
    float a[2];   /* real and imaginary parts of nonzero value */
};

/**
 * `mtx_vector_coordinate_complex_double' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `complex' field, when using double precision data types.
 */
struct mtx_vector_coordinate_complex_double
{
    int i;        /* row index */
    double a[2];   /* real and imaginary parts of nonzero value */
};

/**
 * `mtx_vector_coordinate_integer_single' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `integer' field, when using single precision data types.
 */
struct mtx_vector_coordinate_integer_single
{
    int i;      /* row index */
    int32_t a;  /* nonzero value */
};

/**
 * `mtx_vector_coordinate_integer_double' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `integer' field, when using double precision data types.
 */
struct mtx_vector_coordinate_integer_double
{
    int i;      /* row index */
    int64_t a;  /* nonzero value */
};

/**
 * `mtx_vector_coordinate_pattern' represents a nonzero vector entry
 * in a Matrix Market file with `vector' object, `coordinate' format
 * and `pattern' field.
 */
struct mtx_vector_coordinate_pattern
{
    int i; /* row index */
};

/**
 * `mtx_vector_coordinate_data' is a data structure for representing
 * data associated with vectors in coordinate format.
 */
struct mtx_vector_coordinate_data
{
    /**
     * `field' is the field associated with the vector values: `real',
     * `complex', `integer' or `pattern'.
     */
    enum mtx_field field;

    /**
     * `precision' is the precision associated with the vector values:
     * `single' or `double'.
     */
    enum mtxprecision precision;

    /**
     * `sorting' is the sorting of vector nonzeros: `unsorted',
     * 'row-major' or 'column-major'.
     *
     * Note that the sorting is not explicitly stored in a Matrix
     * Market file, but it is useful additional data that can be
     * provided by the user.
     */
    enum mtx_sorting sorting;

    /**
     * `assembly' is the vector assembly state: `unassembled' or
     * `assembled'.
     *
     * An unassembled sparse vector may contain more than one value
     * associated with each nonzero vector entry. In contrast, there
     * is only one value associated with each nonzero vector entry of
     * an assembled sparse vector.
     *
     * Note that the assembly state is not explicitly stored in a
     * Matrix Market file, but it is useful additional data that can
     * be provided by the user.
     */
    enum mtx_assembly assembly;

    /**
     * `num_rows' is the number of rows in the vector.
     */
    int num_rows;

    /**
     * `num_columns' is the number of columns in the vector.
     */
    int num_columns;

    /**
     * `size' is the number of entries stored in the `data' array.
     */
    int64_t size;

    /**
     * `data' is used to store the vector values.
     *
     * The storage format of nonzero values depends on `field' and
     * `precision'.  Only the member of the `data' union that
     * corresponds to the vector's `field' and `precision' should be
     * used to access the data.
     *
     * For example, if `field' is `real' and `precision' is `single',
     * then `data.real_single' is an array of `size' values of type
     * `struct mtx_vector_coordinate_real_single', which contains the
     * locations and values of the vector entries.
     */
    union {
        struct mtx_vector_coordinate_real_single * real_single;
        struct mtx_vector_coordinate_real_double * real_double;
        struct mtx_vector_coordinate_complex_single * complex_single;
        struct mtx_vector_coordinate_complex_double * complex_double;
        struct mtx_vector_coordinate_integer_single * integer_single;
        struct mtx_vector_coordinate_integer_double * integer_double;
        struct mtx_vector_coordinate_pattern * pattern;
    } data;
};

/**
 * `mtx_vector_coordinate_data_free()' frees resources associated with
 * a vector in coordinate format.
 */
void mtx_vector_coordinate_data_free(
    struct mtx_vector_coordinate_data * mtxdata);

/**
 * `mtx_vector_coordinate_data_alloc()' allocates data for a vector in
 * coordinate format.
 */
int mtx_vector_coordinate_data_alloc(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_field field,
    enum mtxprecision precision,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size);

/*
 * Coordinate vector allocation and initialisation.
 */

/**
 * `mtx_vector_coordinate_data_init_real_single()' creates data for a
 * vector with real, single-precision floating point coefficients.
 */
int mtx_vector_coordinate_data_init_real_single(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_real_single * data);

/**
 * `mtx_vector_coordinate_data_init_real_double()' creates data for a
 * vector with real, double-precision floating point coefficients.
 */
int mtx_vector_coordinate_data_init_real_double(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_real_double * data);

/**
 * `mtx_vector_coordinate_data_init_complex_single()' creates data for
 * a vector with complex, single-precision floating point
 * coefficients.
 */
int mtx_vector_coordinate_data_init_complex_single(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_complex_single * data);

/**
 * `mtx_vector_coordinate_data_init_complex_double()' creates data for
 * a vector with complex, double-precision floating point
 * coefficients.
 */
int mtx_vector_coordinate_data_init_complex_double(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_complex_double * data);

/**
 * `mtx_vector_coordinate_data_init_integer_single()' creates data for
 * a vector with integer, single-precision coefficients.
 */
int mtx_vector_coordinate_data_init_integer_single(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_integer_single * data);

/**
 * `mtx_vector_coordinate_data_init_integer_double()' creates data for
 * a vector with integer, double-precision coefficients.
 */
int mtx_vector_coordinate_data_init_integer_double(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_integer_double * data);

/**
 * `mtx_vector_coordinate_data_init_pattern()' creates data for a
 * vector with boolean (pattern) coefficients.
 */
int mtx_vector_coordinate_data_init_pattern(
    struct mtx_vector_coordinate_data * mtxdata,
    enum mtx_sorting sorting,
    enum mtx_assembly assembly,
    int num_rows,
    int num_columns,
    int64_t size,
    const struct mtx_vector_coordinate_pattern * data);

/**
 * `mtx_vector_coordinate_data_copy_alloc()' allocates a copy of a vector
 * without copying the vector values.
 */
int mtx_vector_coordinate_data_copy_alloc(
    struct mtx_vector_coordinate_data * dst,
    const struct mtx_vector_coordinate_data * src);

/**
 * `mtx_vector_coordinate_data_copy_init()' creates a copy of a vector and
 * also copies vector values.
 */
int mtx_vector_coordinate_data_copy_init(
    struct mtx_vector_coordinate_data * dst,
    const struct mtx_vector_coordinate_data * src);

/**
 * `mtx_vector_coordinate_data_set_zero()' zeroes a vector.
 */
int mtx_vector_coordinate_data_set_zero(
    struct mtx_vector_coordinate_data * mtxdata);

/**
 * `mtx_vector_coordinate_data_set_constant_real_single()' sets every
 * (nonzero) value of a vector equal to a constant, single precision
 * floating point number.
 */
int mtx_vector_coordinate_data_set_constant_real_single(
    struct mtx_vector_coordinate_data * mtxdata,
    float a);

/**
 * `mtx_vector_coordinate_data_set_constant_real_double()' sets every
 * (nonzero) value of a vector equal to a constant, double precision
 * floating point number.
 */
int mtx_vector_coordinate_data_set_constant_real_double(
    struct mtx_vector_coordinate_data * mtxdata,
    double a);

/**
 * `mtx_vector_coordinate_data_set_constant_complex_single()' sets
 * every (nonzero) value of a vector equal to a constant, single
 * precision floating point complex number.
 */
int mtx_vector_coordinate_data_set_constant_complex_single(
    struct mtx_vector_coordinate_data * mtxdata,
    float a[2]);

/**
 * `mtx_vector_coordinate_data_set_constant_complex_double()' sets
 * every (nonzero) value of a vector equal to a constant, double
 * precision floating point complex number.
 */
int mtx_vector_coordinate_data_set_constant_complex_double(
    struct mtx_vector_coordinate_data * mtxdata,
    double a[2]);

/**
 * `mtx_vector_coordinate_data_set_constant_integer_single()' sets
 * every (nonzero) value of a vector equal to a constant integer.
 */
int mtx_vector_coordinate_data_set_constant_integer_single(
    struct mtx_vector_coordinate_data * mtxdata,
    int32_t a);

/**
 * `mtx_vector_coordinate_data_set_constant_integer_double()' sets
 * every (nonzero) value of a vector equal to a constant integer.
 */
int mtx_vector_coordinate_data_set_constant_integer_double(
    struct mtx_vector_coordinate_data * mtxdata,
    int64_t a);

#endif
