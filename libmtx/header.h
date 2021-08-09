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
 * Data types for the Matrix Market header.
 */

#ifndef LIBMTX_HEADER_H
#define LIBMTX_HEADER_H

/*
 * Matrix Market header types.
 */

/**
 * `mtx_object` is used to enumerate different kinds of Matrix Market
 * objects.
 */
enum mtx_object
{
    mtx_matrix,
    mtx_vector
};

/**
 * `mtx_object_str()` is a string representing the Matrix Market
 * object type.
 */
const char * mtx_object_str(
    enum mtx_object object);

/**
 * `mtx_format` is used to enumerate different kinds of Matrix Market
 * formats.
 */
enum mtx_format
{
    mtx_array,     /* array of dense matrix values */
    mtx_coordinate /* coordinate format of sparse matrix values */
};

/**
 * `mtx_format_str()` is a string representing the Matrix Market
 * format type.
 */
const char * mtx_format_str(
    enum mtx_format format);

/**
 * `mtx_field` is used to enumerate different kinds of fields for
 * matrix values in Matrix Market files.
 */
enum mtx_field
{
    mtx_real,    /* single-precision floating point coefficients */
    mtx_double,  /* double-precision floating point coefficients */
    mtx_complex, /* single-precision floating point complex
                  * coefficients */
    mtx_integer, /* integer coefficients */
    mtx_pattern  /* boolean coefficients (sparsity pattern) */
};

/**
 * `mtx_field_str()` is a string representing the Matrix Market field
 * type.
 */
const char * mtx_field_str(
    enum mtx_field field);

/**
 * `mtx_symmetry` is used to enumerate different kinds of symmetry for
 * matrices in Matrix Market format.
 */
enum mtx_symmetry
{
    mtx_general,        /* general, non-symmetric matrix */
    mtx_symmetric,      /* symmetric matrix */
    mtx_skew_symmetric, /* skew-symmetric matrix */
    mtx_hermitian       /* Hermitian matrix */
};

/**
 * `mtx_symmetry_str()` is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtx_symmetry_str(
    enum mtx_symmetry symmetry);

#endif
