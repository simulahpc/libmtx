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
 * Last modified: 2021-08-09
 *
 * Input/output for sparse matrices in coordinate format.
 */

#ifndef LIBMTX_MATRIX_COORDINATE_IO_H
#define LIBMTX_MATRIX_COORDINATE_IO_H

#include <libmtx/mtx/header.h>

#include <stdint.h>

struct mtx_matrix_coordinate_real_single;
struct mtx_matrix_coordinate_real_double;
struct mtx_matrix_coordinate_complex_single;
struct mtx_matrix_coordinate_complex_double;
struct mtx_matrix_coordinate_integer_single;
struct mtx_matrix_coordinate_integer_double;
struct mtx_matrix_coordinate_pattern;

/**
 * `mtx_matrix_coordinate_parse_data_real_single()' parses a data line
 * from a Matrix Market file for a single precision, real matrix in
 * coordinate format.
 */
int mtx_matrix_coordinate_parse_data_real_single(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_real_single * data,
    int num_rows,
    int num_columns);

/**
 * `mtx_matrix_coordinate_parse_data_real_double()' parses a data line
 * from a Matrix Market file for a double precision, real matrix in
 * coordinate format.
 */
int mtx_matrix_coordinate_parse_data_real_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_real_double * data,
    int num_rows,
    int num_columns);

/**
 * `mtx_matrix_coordinate_parse_data_complex_single()' parses a data
 * line from a Matrix Market file for a single precision, complex
 * matrix in coordinate format.
 */
int mtx_matrix_coordinate_parse_data_complex_single(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_complex_single * data,
    int num_rows,
    int num_columns);

/**
 * `mtx_matrix_coordinate_parse_data_complex_double()' parses a data
 * line from a Matrix Market file for a double precision, complex
 * matrix in coordinate format.
 */
int mtx_matrix_coordinate_parse_data_complex_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_complex_double * data,
    int num_rows,
    int num_columns);

/**
 * `mtx_matrix_coordinate_parse_data_integer_single()' parses a data
 * line from a Matrix Market file for a single precision, integer
 * matrix in coordinate format.
 */
int mtx_matrix_coordinate_parse_data_integer_single(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_integer_single * data,
    int num_rows,
    int num_columns);

/**
 * `mtx_matrix_coordinate_parse_data_integer_double()' parses a data
 * line from a Matrix Market file for a double precision, integer
 * matrix in coordinate format.
 */
int mtx_matrix_coordinate_parse_data_integer_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_integer_double * data,
    int num_rows,
    int num_columns);

/**
 * `mtx_matrix_coordinate_parse_data_pattern()' parses a data line from a
 * Matrix Market file for a pattern matrix in coordinate format.
 */
int mtx_matrix_coordinate_parse_data_pattern(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_matrix_coordinate_pattern * data,
    int num_rows,
    int num_columns);

#endif
