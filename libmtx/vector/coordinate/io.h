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
 * Input/output for sparse vectors in coordinate format.
 */

#ifndef LIBMTX_VECTOR_COORDINATE_IO_H
#define LIBMTX_VECTOR_COORDINATE_IO_H

#include <libmtx/mtx/header.h>

#include <stdint.h>

struct mtx;

/**
 * `mtx_vector_coordinate_parse_size()` parses a size line from a
 * Matrix Market file for a vector in coordinate format.
 */
int mtx_vector_coordinate_parse_size(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    enum mtx_object object,
    enum mtx_format format,
    enum mtx_field field,
    enum mtx_symmetry symmetry,
    int * num_rows,
    int * num_columns,
    int64_t * num_nonzeros,
    int64_t * size,
    int * nonzero_size);

/**
 * `mtx_vector_coordinate_parse_data_real()' parses a data line from a
 * Matrix Market file for a real vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_real(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_real * data,
    int num_rows);

/**
 * `mtx_vector_coordinate_parse_data_double()' parses a data line from a
 * Matrix Market file for a double vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_double(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_double * data,
    int num_rows);

/**
 * `mtx_vector_coordinate_parse_data_complex()' parses a data line from a
 * Matrix Market file for a complex vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_complex(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_complex * data,
    int num_rows);

/**
 * `mtx_vector_coordinate_parse_data_integer()' parses a data line from a
 * Matrix Market file for a integer vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_integer(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_integer * data,
    int num_rows);

/**
 * `mtx_vector_coordinate_parse_data_pattern()' parses a data line from a
 * Matrix Market file for a pattern vector in coordinate format.
 */
int mtx_vector_coordinate_parse_data_pattern(
    const char * line,
    int * bytes_read,
    const char ** endptr,
    struct mtx_vector_coordinate_pattern * data,
    int num_rows);

#endif
