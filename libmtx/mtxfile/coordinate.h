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
 * Last modified: 2021-09-01
 *
 * Matrix Market coordinate data.
 */

#ifndef LIBMTX_MTXFILE_COORDINATE_H
#define LIBMTX_MTXFILE_COORDINATE_H

#include <stdint.h>

/*
 * Matrix coordinate formats
 */

/**
 * `mtxfile_matrix_coordinate_real_single' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `real' field, when using single precision data types.
 */
struct mtxfile_matrix_coordinate_real_single
{
    int i;    /* row index */
    int j;    /* column index */
    float a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_double' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `real' field, when using double precision data types.
 */
struct mtxfile_matrix_coordinate_real_double
{
    int i;     /* row index */
    int j;     /* column index */
    double a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_complex_single' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `complex' field, when using single
 * precision data types.
 */
struct mtxfile_matrix_coordinate_complex_single
{
    int i;       /* row index */
    int j;       /* column index */
    float a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_complex_double' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `complex' field, when using double
 * precision data types.
 */
struct mtxfile_matrix_coordinate_complex_double
{
    int i;       /* row index */
    int j;       /* column index */
    double a[2];  /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_integer_single' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `integer' field, when using single
 * precision data types.
 */
struct mtxfile_matrix_coordinate_integer_single
{
    int i;      /* row index */
    int j;      /* column index */
    int32_t a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_integer_double' represents a nonzero
 * matrix entry in a Matrix Market file with `matrix' object,
 * `coordinate' format and `integer' field, when using double
 * precision data types.
 */
struct mtxfile_matrix_coordinate_integer_double
{
    int i;      /* row index */
    int j;      /* column index */
    int64_t a;  /* nonzero value */
};

/**
 * `mtxfile_matrix_coordinate_pattern' represents a nonzero matrix
 * entry in a Matrix Market file with `matrix' object, `coordinate'
 * format and `pattern' field.
 */
struct mtxfile_matrix_coordinate_pattern
{
    int i;  /* row index */
    int j;  /* column index */
};

/*
 * Vector coordinate formats
 */

/**
 * `mtxfile_vector_coordinate_real_single' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `real' field, when using single precision data types.
 */
struct mtxfile_vector_coordinate_real_single
{
    int i;    /* row index */
    float a;  /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_double' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `real' field, when using double precision data types.
 */
struct mtxfile_vector_coordinate_real_double
{
    int i;    /* row index */
    double a; /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_complex_single' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `complex' field, when using single
 * precision data types.
 */
struct mtxfile_vector_coordinate_complex_single
{
    int i;        /* row index */
    float a[2];   /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_vector_coordinate_complex_double' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `complex' field, when using double
 * precision data types.
 */
struct mtxfile_vector_coordinate_complex_double
{
    int i;        /* row index */
    double a[2];   /* real and imaginary parts of nonzero value */
};

/**
 * `mtxfile_vector_coordinate_integer_single' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `integer' field, when using single
 * precision data types.
 */
struct mtxfile_vector_coordinate_integer_single
{
    int i;      /* row index */
    int32_t a;  /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_integer_double' represents a nonzero
 * vector entry in a Matrix Market file with `vector' object,
 * `coordinate' format and `integer' field, when using double
 * precision data types.
 */
struct mtxfile_vector_coordinate_integer_double
{
    int i;      /* row index */
    int64_t a;  /* nonzero value */
};

/**
 * `mtxfile_vector_coordinate_pattern' represents a nonzero vector
 * entry in a Matrix Market file with `vector' object, `coordinate'
 * format and `pattern' field.
 */
struct mtxfile_vector_coordinate_pattern
{
    int i; /* row index */
};

#endif
