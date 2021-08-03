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
 * Last modified: 2021-06-18
 *
 * Data types for the Matrix Market header.
 */

#ifndef MATRIXMARKET_HEADER_H
#define MATRIXMARKET_HEADER_H

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

/*
 * Additional types used with objects in Matrix Market format.
 */

/**
 * `mtx_triangle' is used to enumerate matrix properties related to
 * whether or not matrices are upper or lower triangular. Note that
 * the term triangular is still used even for non-square matrices,
 * where the term trapezoidal would be more accurate.
 */
enum mtx_triangle {
    mtx_nontriangular,           /* nonzero above, below or on main diagonal */
    mtx_lower_triangular,        /* zero above main diagonal */
    mtx_upper_triangular,        /* zero below main diagonal */
    mtx_diagonal,                /* zero above and below main diagonal */
    /* mtx_unit_lower_triangular,   /\* one on main diagonal and zero above *\/ */
    /* mtx_unit_upper_triangular,   /\* one on main diagonal and zero below *\/ */
    /* mtx_unit_diagonal,           /\* one on main diagonal, zero above and below *\/ */
    /* mtx_strict_lower_triangular, /\* zero on or above main diagonal *\/ */
    /* mtx_strict_upper_triangular, /\* zero on or below main diagonal *\/ */
    /* mtx_zero,                    /\* zero on, above and below main diagonal *\/ */
};

/**
 * `mtx_triangle_str()' is a string representing the given triangle
 * type.
 */
const char * mtx_triangle_str(
    enum mtx_triangle triangle);

/**
 * `mtx_sorting` is used to enumerate different kinds of sortings of
 * matrix nonzeros for matrices in Matrix Market format.
 */
enum mtx_sorting
{
    mtx_unsorted,       /* unsorted matrix nonzeros */
    mtx_row_major,      /* row major ordering */
    mtx_column_major,   /* column major ordering */
};

/**
 * `mtx_sorting_str()` is a string representing the sorting type.
 */
const char * mtx_sorting_str(
    enum mtx_sorting sorting);

/**
 * `mtx_ordering` is used to enumerate different kinds of orderings
 * for matrices in Matrix Market format.
 */
enum mtx_ordering
{
    mtx_unordered,      /* general, unordered matrix */
    mtx_rcm,            /* Reverse Cuthill-McKee ordering */
};

/**
 * `mtx_ordering_str()` is a string representing the ordering type.
 */
const char * mtx_ordering_str(
    enum mtx_ordering ordering);

/**
 * `mtx_assembly` is used to enumerate assembly states for sparse
 * matrices in Matrix Market format.
 */
enum mtx_assembly
{
    mtx_unassembled,    /* unassembled; duplicate nonzeros allowed. */
    mtx_assembled,      /* assembled; duplicate nonzeros not allowed. */
};

/**
 * `mtx_assembly_str()` is a string representing the assembly type.
 */
const char * mtx_assembly_str(
    enum mtx_assembly assembly);

/**
 * `mtx_partitioning` is used to enumerate different ways of
 * partitioning matrices and vectors in Matrix Market format.
 */
enum mtx_partitioning
{
    mtx_partition,   /* matrix/vector entries are owned by a single
                      * MPI process. */
    mtx_cover,       /* matrix/vector entries may be shared by
                      * multiple MPI processes. */
};

/**
 * `mtx_partitioning_str()` is a string representing the partitioning
 * type.
 */
const char * mtx_partitioning_str(
    enum mtx_partitioning partitioning);

#endif
