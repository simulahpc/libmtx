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

#include <libmtx/header.h>

/**
 * `mtx_object_str()` is a string representing the Matrix Market object
 * type.
 */
const char * mtx_object_str(
    enum mtx_object object)
{
    switch (object) {
    case mtx_matrix: return "matrix";
    case mtx_vector: return "vector";
    default: return "unknown";
    }
}

/**
 * `mtx_format_str()` is a string representing the Matrix Market format
 * type.
 */
const char * mtx_format_str(
    enum mtx_format format)
{
    switch (format) {
    case mtx_array: return "array";
    case mtx_coordinate: return "coordinate";
    default: return "unknown";
    }
}

/**
 * `mtx_field_str()` is a string representing the Matrix Market field
 * type.
 */
const char * mtx_field_str(
    enum mtx_field field)
{
    switch (field) {
    case mtx_real: return "real";
    case mtx_double: return "double";
    case mtx_complex: return "complex";
    case mtx_integer: return "integer";
    case mtx_pattern: return "pattern";
    default: return "unknown";
    }
}

/**
 * `mtx_symmetry_str()` is a string representing the Matrix Market
 * symmetry type.
 */
const char * mtx_symmetry_str(
    enum mtx_symmetry symmetry)
{
    switch (symmetry) {
    case mtx_general: return "general";
    case mtx_symmetric: return "symmetric";
    case mtx_skew_symmetric: return "skew-symmetric";
    case mtx_hermitian: return "hermitian";
    default: return "unknown";
    }
}

/**
 * `mtx_triangle_str()' is a string representing the given triangle
 * type.
 */
const char * mtx_triangle_str(
    enum mtx_triangle triangle)
{
    switch (triangle) {
    case mtx_nontriangular: return "nontriangular";
    case mtx_lower_triangular: return "lower-triangular";
    case mtx_upper_triangular: return "upper-triangular";
    case mtx_diagonal: return "diagonal";
    /* case mtx_unit_lower_triangular: return "unit-lower-triangular"; */
    /* case mtx_unit_upper_triangular: return "unit-upper-triangular"; */
    /* case mtx_unit_diagonal: return "unit-diagonal"; */
    /* case mtx_strict_lower_triangular: return "strict-lower-triangular"; */
    /* case mtx_strict_upper_triangular: return "strict-upper-triangular"; */
    /* case mtx_zero: return "zero"; */
    default: return "unknown";
    }
}

/**
 * `mtx_sorting_str()` is a string representing the sorting type.
 */
const char * mtx_sorting_str(
    enum mtx_sorting sorting)
{
    switch (sorting) {
    case mtx_unsorted: return "unsorted";
    case mtx_row_major: return "row-major";
    case mtx_column_major: return "column-major";
    default: return "unknown";
    }
}

/**
 * `mtx_ordering_str()` is a string representing the ordering type.
 */
const char * mtx_ordering_str(
    enum mtx_ordering ordering)
{
    switch (ordering) {
    case mtx_unordered: return "unordered";
    case mtx_rcm: return "rcm";
    default: return "unknown";
    }
}

/**
 * `mtx_assembly_str()` is a string representing the assembly type.
 */
const char * mtx_assembly_str(
    enum mtx_assembly assembly)
{
    switch (assembly) {
    case mtx_unassembled: return "unassembled";
    case mtx_assembled: return "assembled";
    default: return "unknown";
    }
}

/**
 * `mtx_partitioning_str()` is a string representing the partitioning
 * type.
 */
const char * mtx_partitioning_str(
    enum mtx_partitioning partitioning)
{
    switch (partitioning) {
    case mtx_partition: return "partition";
    case mtx_cover: return "cover";
    default: return "unknown";
    }
}
