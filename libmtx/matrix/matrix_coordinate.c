/* This file is part of Libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-03-15
 *
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/matrix_coordinate.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>
#include <libmtx/vector/vector.h>
#include <libmtx/vector/vector_array.h>

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Memory management
 */

/**
 * ‘mtxmatrix_coordinate_free()’ frees storage allocated for a matrix.
 */
void mtxmatrix_coordinate_free(
    struct mtxmatrix_coordinate * matrix)
{
    if (matrix->field == mtx_field_real) {
        if (matrix->precision == mtx_single) {
            free(matrix->data.real_single);
        } else if (matrix->precision == mtx_double) {
            free(matrix->data.real_double);
        }
    } else if (matrix->field == mtx_field_complex) {
        if (matrix->precision == mtx_single) {
            free(matrix->data.complex_single);
        } else if (matrix->precision == mtx_double) {
            free(matrix->data.complex_double);
        }
    } else if (matrix->field == mtx_field_integer) {
        if (matrix->precision == mtx_single) {
            free(matrix->data.integer_single);
        } else if (matrix->precision == mtx_double) {
            free(matrix->data.integer_double);
        }
    }
    free(matrix->colidx);
    free(matrix->rowidx);
}

/**
 * ‘mtxmatrix_coordinate_alloc_copy()’ allocates a copy of a matrix
 * without initialising the values.
 */
int mtxmatrix_coordinate_alloc_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src)
{
    return mtxmatrix_coordinate_alloc(
        dst, src->field, src->precision, src->symmetry,
        src->num_rows, src->num_columns, src->size);
}

/**
 * ‘mtxmatrix_coordinate_init_copy()’ allocates a copy of a matrix and
 * also copies the values.
 */
int mtxmatrix_coordinate_init_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src)
{
    int err = mtxmatrix_coordinate_alloc_copy(dst, src);
    if (err) return err;
    dst->num_nonzeros = src->num_nonzeros;
    err = mtxmatrix_coordinate_copy(dst, src);
    if (err) return err;
    return MTX_SUCCESS;
}

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxmatrix_coordinate_alloc()’ allocates a matrix in coordinate
 * format.
 */
int mtxmatrix_coordinate_alloc(
    struct mtxmatrix_coordinate * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size)
{
    int64_t num_entries;
    if (__builtin_mul_overflow(num_rows, num_columns, &num_entries)) {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }

    matrix->rowidx = malloc(size * sizeof(int));
    if (!matrix->rowidx)
        return MTX_ERR_ERRNO;
    matrix->colidx = malloc(size * sizeof(int));
    if (!matrix->colidx) {
        free(matrix->rowidx);
        return MTX_ERR_ERRNO;
    }
    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            matrix->data.real_single =
                malloc(size * sizeof(*matrix->data.real_single));
            if (!matrix->data.real_single) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            matrix->data.real_double =
                malloc(size * sizeof(*matrix->data.real_double));
            if (!matrix->data.real_double) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else {
            free(matrix->colidx);
            free(matrix->rowidx);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            matrix->data.complex_single =
                malloc(size * sizeof(*matrix->data.complex_single));
            if (!matrix->data.complex_single) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            matrix->data.complex_double =
                malloc(size * sizeof(*matrix->data.complex_double));
            if (!matrix->data.complex_double) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else {
            free(matrix->colidx);
            free(matrix->rowidx);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            matrix->data.integer_single =
                malloc(size * sizeof(*matrix->data.integer_single));
            if (!matrix->data.integer_single) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            matrix->data.integer_double =
                malloc(size * sizeof(*matrix->data.integer_double));
            if (!matrix->data.integer_double) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else {
            free(matrix->colidx);
            free(matrix->rowidx);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_pattern) {
        /* No data needs to be allocated. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    matrix->field = field;
    matrix->precision = precision;
    matrix->symmetry = symmetry;
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->num_entries = num_entries;
    matrix->num_nonzeros = -1;
    matrix->size = size;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_real_single()’ allocates and initialises
 * a matrix in coordinate format with real, single precision
 * coefficients.
 */
int mtxmatrix_coordinate_init_real_single(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_real, mtx_single, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.real_single[k] = data[k];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_real_double()’ allocates and initialises
 * a matrix in coordinate format with real, double precision
 * coefficients.
 */
int mtxmatrix_coordinate_init_real_double(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_real, mtx_double, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.real_double[k] = data[k];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_complex_single()’ allocates and
 * initialises a matrix in coordinate format with complex, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_complex_single(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_complex, mtx_single, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.complex_single[k][0] = data[k][0];
        matrix->data.complex_single[k][1] = data[k][1];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_complex_double()’ allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_complex_double(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_complex, mtx_double, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.complex_double[k][0] = data[k][0];
        matrix->data.complex_double[k][1] = data[k][1];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_integer_single()’ allocates and
 * initialises a matrix in coordinate format with integer, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_integer_single(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_integer, mtx_single, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.integer_single[k] = data[k];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_integer_double()’ allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_integer_double(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_integer, mtx_double, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.integer_double[k] = data[k];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_init_pattern()’ allocates and initialises a
 * matrix in coordinate format with boolean coefficients.
 */
int mtxmatrix_coordinate_init_pattern(
    struct mtxmatrix_coordinate * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t size,
    const int * rowidx,
    const int * colidx)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_pattern, mtx_single, symmetry,
        num_rows, num_columns, size);
    if (err) return err;
    matrix->num_nonzeros = 0;
    for (int64_t k = 0; k < size; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->num_nonzeros +=
            (symmetry == mtx_unsymmetric || rowidx[k] == colidx[k]) ? 1 : 2;
    }
    return MTX_SUCCESS;
}

/*
 * Row and column vectors
 */

/**
 * ‘mtxmatrix_coordinate_alloc_row_vector()’ allocates a row vector
 * for a given matrix, where a row vector is a vector whose length
 * equal to a single row of the matrix.
 */
int mtxmatrix_coordinate_alloc_row_vector(
    const struct mtxmatrix_coordinate * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type)
{
    if (vector_type == mtxvector_auto)
        vector_type = mtxvector_array;

    if (vector_type == mtxvector_array) {
        return mtxvector_alloc_array(
            vector, matrix->field, matrix->precision, matrix->num_columns);
    } else if (vector_type == mtxvector_coordinate) {
        /* TODO: Here we may wish to only allocate a vector with one
         * nonzero for each column of the matrix that contains a
         * nonzero entry. */
        return mtxvector_alloc_coordinate(
            vector, matrix->field, matrix->precision,
            matrix->num_columns, matrix->num_columns);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * ‘mtxmatrix_coordinate_alloc_column_vector()’ allocates a column
 * vector for a given matrix, where a column vector is a vector whose
 * length equal to a single column of the matrix.
 */
int mtxmatrix_coordinate_alloc_column_vector(
    const struct mtxmatrix_coordinate * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type)
{
    if (vector_type == mtxvector_auto)
        vector_type = mtxvector_array;

    if (vector_type == mtxvector_array) {
        return mtxvector_alloc_array(
            vector, matrix->field, matrix->precision, matrix->num_rows);
    } else if (vector_type == mtxvector_coordinate) {
        /* TODO: Here we may wish to only allocate a vector with one
         * nonzero for each row of the matrix that contains a nonzero
         * entry. */
        return mtxvector_alloc_coordinate(
            vector, matrix->field, matrix->precision,
            matrix->num_rows, matrix->num_rows);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmatrix_coordinate_from_mtxfile()’ converts a matrix in Matrix
 * Market format to a coordinate matrix.
 */
int mtxmatrix_coordinate_from_mtxfile(
    struct mtxmatrix_coordinate * matrix,
    const struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from array to coordinate. */
    if (mtxfile->header.format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxsymmetry symmetry;
    err = mtxfilesymmetry_to_mtxsymmetry(
        &symmetry, mtxfile->header.symmetry);
    if (err) return err;

    int num_rows = mtxfile->size.num_rows;
    int num_columns = mtxfile->size.num_columns;
    int64_t size = mtxfile->size.num_nonzeros;

    if (mtxfile->header.field == mtxfile_real) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_real, mtxfile->precision, symmetry,
            num_rows, num_columns, size);
        if (err) return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            matrix->num_nonzeros = 0;
            for (int64_t k = 0; k < size; k++) {
                matrix->rowidx[k] = data[k].i-1;
                matrix->colidx[k] = data[k].j-1;
                matrix->data.real_single[k] = data[k].a;
                matrix->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            matrix->num_nonzeros = 0;
            for (int64_t k = 0; k < size; k++) {
                matrix->rowidx[k] = data[k].i-1;
                matrix->colidx[k] = data[k].j-1;
                matrix->data.real_double[k] = data[k].a;
                matrix->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_complex, mtxfile->precision, symmetry,
            num_rows, num_columns, size);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            matrix->num_nonzeros = 0;
            for (int64_t k = 0; k < size; k++) {
                matrix->rowidx[k] = data[k].i-1;
                matrix->colidx[k] = data[k].j-1;
                matrix->data.complex_single[k][0] = data[k].a[0];
                matrix->data.complex_single[k][1] = data[k].a[1];
                matrix->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            matrix->num_nonzeros = 0;
            for (int64_t k = 0; k < size; k++) {
                matrix->rowidx[k] = data[k].i-1;
                matrix->colidx[k] = data[k].j-1;
                matrix->data.complex_double[k][0] = data[k].a[0];
                matrix->data.complex_double[k][1] = data[k].a[1];
                matrix->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_integer, mtxfile->precision, symmetry,
            num_rows, num_columns, size);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            matrix->num_nonzeros = 0;
            for (int64_t k = 0; k < size; k++) {
                matrix->rowidx[k] = data[k].i-1;
                matrix->colidx[k] = data[k].j-1;
                matrix->data.integer_single[k] = data[k].a;
                matrix->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            matrix->num_nonzeros = 0;
            for (int64_t k = 0; k < size; k++) {
                matrix->rowidx[k] = data[k].i-1;
                matrix->colidx[k] = data[k].j-1;
                matrix->data.integer_double[k] = data[k].a;
                matrix->num_nonzeros +=
                    (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_pattern, mtx_single, symmetry,
            num_rows, num_columns, size);
        if (err)
            return err;
        const struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
            matrix->num_nonzeros = 0;
        for (int64_t k = 0; k < size; k++) {
            matrix->rowidx[k] = data[k].i-1;
            matrix->colidx[k] = data[k].j-1;
            matrix->num_nonzeros +=
                (symmetry == mtx_unsymmetric || data[k].i == data[k].j) ? 1 : 2;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_to_mtxfile()’ converts a coordinate matrix to
 * a matrix in Matrix Market format.
 */
int mtxmatrix_coordinate_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_coordinate * matrix,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxfilesymmetry symmetry;
    err = mtxfilesymmetry_from_mtxsymmetry(&symmetry, matrix->symmetry);
    if (err) return err;

    if (matrix->field == mtx_field_real) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_real, symmetry, matrix->precision,
            matrix->num_rows, matrix->num_columns, matrix->size);
        if (err) return err;
        if (matrix->precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int64_t k = 0; k < matrix->size; k++) {
                data[k].i = matrix->rowidx[k]+1;
                data[k].j = matrix->colidx[k]+1;
                data[k].a = matrix->data.real_single[k];
            }
        } else if (matrix->precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int64_t k = 0; k < matrix->size; k++) {
                data[k].i = matrix->rowidx[k]+1;
                data[k].j = matrix->colidx[k]+1;
                data[k].a = matrix->data.real_double[k];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_complex) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_complex, symmetry, matrix->precision,
            matrix->num_rows, matrix->num_columns, matrix->size);
        if (err) return err;
        if (matrix->precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int64_t k = 0; k < matrix->size; k++) {
                data[k].i = matrix->rowidx[k]+1;
                data[k].j = matrix->colidx[k]+1;
                data[k].a[0] = matrix->data.complex_single[k][0];
                data[k].a[1] = matrix->data.complex_single[k][1];
            }
        } else if (matrix->precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int64_t k = 0; k < matrix->size; k++) {
                data[k].i = matrix->rowidx[k]+1;
                data[k].j = matrix->colidx[k]+1;
                data[k].a[0] = matrix->data.complex_double[k][0];
                data[k].a[1] = matrix->data.complex_double[k][1];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_integer) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_integer, symmetry, matrix->precision,
            matrix->num_rows, matrix->num_columns, matrix->size);
        if (err) return err;
        if (matrix->precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int64_t k = 0; k < matrix->size; k++) {
                data[k].i = matrix->rowidx[k]+1;
                data[k].j = matrix->colidx[k]+1;
                data[k].a = matrix->data.integer_single[k];
            }
        } else if (matrix->precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int64_t k = 0; k < matrix->size; k++) {
                data[k].i = matrix->rowidx[k]+1;
                data[k].j = matrix->colidx[k]+1;
                data[k].a = matrix->data.integer_double[k];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_pattern) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_pattern, symmetry, mtx_single,
            matrix->num_rows, matrix->num_columns, matrix->size);
        if (err) return err;
        struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < matrix->size; k++) {
            data[k].i = matrix->rowidx[k]+1;
            data[k].j = matrix->colidx[k]+1;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/*
 * Nonzero rows and columns
 */

/**
 * ‘mtxmatrix_coordinate_nzrows()’ counts the number of nonzero
 * (non-empty) matrix rows, and, optionally, fills an array with the
 * row indices of the nonzero (non-empty) matrix rows.
 *
 * If ‘num_nonzero_rows’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix rows.
 *
 * ‘nonzero_rows’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the row
 * indices of the nonzero matrix rows. Note that ‘size’ must be at
 * least equal to the number of non-zero rows.
 */
int mtxmatrix_coordinate_nzrows(
    const struct mtxmatrix_coordinate * matrix,
    int * num_nonzero_rows,
    int size,
    int * nonzero_rows)
{
    int err;
    int * rowidx = malloc(matrix->size * sizeof(int));
    if (!rowidx) return MTX_ERR_ERRNO;
    for (int64_t k = 0; k < matrix->size; k++)
        rowidx[k] = matrix->rowidx[k];

    /* sort, then compact the sorted row indices */
    err = radix_sort_int(size, rowidx, NULL);
    if (err) {
        free(rowidx);
        return err;
    }
    int n = 0;
    for (int64_t k = 1; k < matrix->size; k++) {
        if (rowidx[n] != rowidx[k])
            rowidx[++n] = rowidx[k];
    }
    n = matrix->size == 0 ? 0 : n+1;
    if (num_nonzero_rows) *num_nonzero_rows = n;
    if (nonzero_rows) {
        if (size < n) {
            free(rowidx);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
        for (int i = 0; i < n; i++)
            nonzero_rows[i] = rowidx[i];
    }
    free(rowidx);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_nzcols()’ counts the number of nonzero
 * (non-empty) matrix columns, and, optionally, fills an array with
 * the column indices of the nonzero (non-empty) matrix columns.
 *
 * If ‘num_nonzero_columns’ is ‘NULL’, then it is ignored, or else it
 * must point to an integer that is used to store the number of
 * nonzero matrix columns.
 *
 * ‘nonzero_columns’ may be ‘NULL’, in which case it is ignored.
 * Otherwise, it must point to an array of length at least equal to
 * ‘size’. On successful completion, this array contains the column
 * indices of the nonzero matrix columns. Note that ‘size’ must be at
 * least equal to the number of non-zero columns.
 */
int mtxmatrix_coordinate_nzcols(
    const struct mtxmatrix_coordinate * matrix,
    int * num_nonzero_columns,
    int size,
    int * nonzero_columns)
{
    int err;
    int * colidx = malloc(matrix->size * sizeof(int));
    if (!colidx) return MTX_ERR_ERRNO;
    for (int64_t k = 0; k < matrix->size; k++)
        colidx[k] = matrix->colidx[k];

    /* sort, then compact the sorted column indices */
    err = radix_sort_int(size, colidx, NULL);
    if (err) {
        free(colidx);
        return err;
    }
    int n = 0;
    for (int64_t k = 1; k < matrix->size; k++) {
        if (colidx[n] != colidx[k])
            colidx[++n] = colidx[k];
    }
    n = matrix->size == 0 ? 0 : n+1;
    if (num_nonzero_columns) *num_nonzero_columns = n;
    if (nonzero_columns) {
        if (size < n) {
            free(colidx);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
        for (int i = 0; i < n; i++)
            nonzero_columns[i] = colidx[i];
    }
    free(colidx);
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxmatrix_coordinate_partition()’ partitions a matrix into blocks
 * according to the given row and column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix ‘src’, respectively. That is, ‘rowpart->size’
 * must be equal to the number of matrix rows, and ‘colpart->size’
 * must be equal to the number of matrix columns.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxmatrix’, where ‘P’ is the number of
 * row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * matrix in the ‘dsts’ array.
 */
int mtxmatrix_coordinate_partition(
    struct mtxmatrix * dsts,
    const struct mtxmatrix_coordinate * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    int err;
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    int num_parts = num_row_parts * num_col_parts;

    struct mtxfile mtxfile;
    err = mtxmatrix_coordinate_to_mtxfile(&mtxfile, src, mtxfile_coordinate);
    if (err) return err;

    struct mtxfile * dstmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!dstmtxfiles) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_ERRNO;
    }

    err = mtxfile_partition(dstmtxfiles, &mtxfile, rowpart, colpart);
    if (err) {
        free(dstmtxfiles);
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);

    for (int p = 0; p < num_parts; p++) {
        dsts[p].type = mtxmatrix_coordinate;
        err = mtxmatrix_coordinate_from_mtxfile(
            &dsts[p].storage.coordinate, &dstmtxfiles[p]);
        if (err) {
            for (int q = p; q < num_parts; q++)
                mtxfile_free(&dstmtxfiles[q]);
            free(dstmtxfiles);
            return err;
        }
        mtxfile_free(&dstmtxfiles[p]);
    }
    free(dstmtxfiles);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_join()’ joins together matrices representing
 * compatible blocks of a partitioned matrix to form a larger matrix.
 *
 * The argument ‘srcs’ is logically arranged as a two-dimensional
 * array of size ‘P*Q’, where ‘P’ is the number of row parts
 * (‘rowpart->num_parts’) and ‘Q’ is the number of column parts
 * (‘colpart->num_parts’).  Note that the ‘r’th part corresponds to a
 * row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus, the
 * ‘r’th entry of ‘srcs’ is the submatrix corresponding to the ‘p’th
 * row and ‘q’th column of the 2D partitioning.
 *
 * Moreover, the blocks must be compatible, which means that each part
 * in the same block row ‘p’, must have the same number of rows.
 * Similarly, each part in the same block column ‘q’ must have the
 * same number of columns. Finally, for each block column ‘q’, the sum
 * of the number of rows of ‘srcs[p*Q+q]’ for ‘p=0,1,...,P-1’ must be
 * equal to ‘rowpart->size’. Likewise, for each block row ‘p’, the sum
 * of the number of columns of ‘srcs[p*Q+q]’ for ‘q=0,1,...,Q-1’ must
 * be equal to ‘colpart->size’.
 */
int mtxmatrix_coordinate_join(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix * srcs,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    int err;
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    int num_parts = num_row_parts * num_col_parts;

    struct mtxfile * srcmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!srcmtxfiles) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        err = mtxmatrix_to_mtxfile(&srcmtxfiles[p], &srcs[p], mtxfile_coordinate);
        if (err) {
            for (int q = p-1; q >= 0; q--)
                mtxfile_free(&srcmtxfiles[q]);
            free(srcmtxfiles);
            return err;
        }
    }

    struct mtxfile dstmtxfile;
    err = mtxfile_join(&dstmtxfile, srcmtxfiles, rowpart, colpart);
    if (err) {
        for (int p = 0; p < num_parts; p++)
            mtxfile_free(&srcmtxfiles[p]);
        free(srcmtxfiles);
        return err;
    }
    for (int p = 0; p < num_parts; p++)
        mtxfile_free(&srcmtxfiles[p]);
    free(srcmtxfiles);

    err = mtxmatrix_coordinate_from_mtxfile(dst, &dstmtxfile);
    if (err) {
        mtxfile_free(&dstmtxfile);
        return err;
    }
    mtxfile_free(&dstmtxfile);
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

static bool same_pattern(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y)
{
    return x->symmetry == y->symmetry
        && x->num_rows == y->num_rows
        && x->num_columns == y->num_columns
        && x->size == y->size
        && memcmp(x->rowidx, y->rowidx, x->size*sizeof(*x->rowidx)) == 0
        && memcmp(x->colidx, y->colidx, x->size*sizeof(*x->colidx)) == 0;
}

static int vectorise_array(
    struct mtxvector_array * vecx,
    const struct mtxmatrix_coordinate * x)
{
    vecx->field = x->field;
    vecx->precision = x->precision;
    vecx->size = x->size;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            vecx->data.real_single = x->data.real_single;
        } else if (x->precision == mtx_double) {
            vecx->data.real_double = x->data.real_double;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            vecx->data.complex_single = x->data.complex_single;
        } else if (x->precision == mtx_double) {
            vecx->data.complex_double = x->data.complex_double;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            vecx->data.integer_single = x->data.integer_single;
        } else if (x->precision == mtx_double) {
            vecx->data.integer_double = x->data.integer_double;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_swap(
    struct mtxmatrix_coordinate * x,
    struct mtxmatrix_coordinate * y)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    struct mtxvector_array vecx;
    int err = vectorise_array(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = vectorise_array(&vecy, x);
    if (err) return err;
    return mtxvector_array_swap(&vecx, &vecy);
}

/**
 * ‘mtxmatrix_coordinate_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_copy(
    struct mtxmatrix_coordinate * y,
    const struct mtxmatrix_coordinate * x)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    struct mtxvector_array vecy;
    int err = vectorise_array(&vecy, y);
    if (err) return err;
    struct mtxvector_array vecx;
    err = vectorise_array(&vecx, x);
    if (err) return err;
    return mtxvector_array_copy(&vecy, &vecx);
}

/**
 * ‘mtxmatrix_coordinate_sscal()’ scales a matrix by a single
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_coordinate_sscal(
    float a,
    struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = vectorise_array(&vecx, x);
    if (err) return err;
    return mtxvector_array_sscal(a, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_dscal()’ scales a matrix by a double
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_coordinate_dscal(
    double a,
    struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = vectorise_array(&vecx, x);
    if (err) return err;
    return mtxvector_array_dscal(a, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_saxpy()’ adds a matrix to another one
 * multiplied by a single precision floating point value, ‘y = a*x +
 * y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_saxpy(
    float a,
    const struct mtxmatrix_coordinate * x,
    struct mtxmatrix_coordinate * y,
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    struct mtxvector_array vecx;
    int err = vectorise_array(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = vectorise_array(&vecy, y);
    if (err) return err;
    return mtxvector_array_saxpy(a, &vecx, &vecy, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_daxpy()’ adds a matrix to another one
 * multiplied by a double precision floating point value, ‘y = a*x +
 * y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_daxpy(
    double a,
    const struct mtxmatrix_coordinate * x,
    struct mtxmatrix_coordinate * y,
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    struct mtxvector_array vecx;
    int err = vectorise_array(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = vectorise_array(&vecy, y);
    if (err) return err;
    return mtxvector_array_daxpy(a, &vecx, &vecy, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_saypx()’ multiplies a matrix by a single
 * precision floating point scalar and adds another matrix, ‘y = a*y +
 * x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_saypx(
    float a,
    struct mtxmatrix_coordinate * y,
    const struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    struct mtxvector_array vecy;
    int err = vectorise_array(&vecy, y);
    if (err) return err;
    struct mtxvector_array vecx;
    err = vectorise_array(&vecx, x);
    if (err) return err;
    return mtxvector_array_saypx(a, &vecy, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_daypx()’ multiplies a matrix by a double
 * precision floating point scalar and adds another matrix, ‘y = a*y +
 * x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_daypx(
    double a,
    struct mtxmatrix_coordinate * y,
    const struct mtxmatrix_coordinate * x,
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    struct mtxvector_array vecy;
    int err = vectorise_array(&vecy, y);
    if (err) return err;
    struct mtxvector_array vecx;
    err = vectorise_array(&vecx, x);
    if (err) return err;
    return mtxvector_array_daypx(a, &vecy, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_coordinate_sdot()’ computes the Frobenius inner product
 * of two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_sdot(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    float * dot,
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        struct mtxvector_array vecy;
        err = vectorise_array(&vecy, y);
        if (err) return err;
        return mtxvector_array_sdot(&vecx, &vecy, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_ddot()’ computes the Frobenius inner product
 * of two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_ddot(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    double * dot,
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        struct mtxvector_array vecy;
        err = vectorise_array(&vecy, y);
        if (err) return err;
        return mtxvector_array_ddot(&vecx, &vecy, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_cdotu()’ computes the product of the
 * transpose of a complex row matrix with another complex row matrix
 * in single precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_cdotu(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        struct mtxvector_array vecy;
        err = vectorise_array(&vecy, y);
        if (err) return err;
        return mtxvector_array_cdotu(&vecx, &vecy, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_zdotu()’ computes the product of the
 * transpose of a complex row matrix with another complex row matrix
 * in double precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_zdotu(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        struct mtxvector_array vecy;
        err = vectorise_array(&vecy, y);
        if (err) return err;
        return mtxvector_array_zdotu(&vecx, &vecy, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_cdotc()’ computes the Frobenius inner product
 * of two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_cdotc(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        struct mtxvector_array vecy;
        err = vectorise_array(&vecy, y);
        if (err) return err;
        return mtxvector_array_cdotc(&vecx, &vecy, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_zdotc()’ computes the Frobenius inner product
 * of two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_coordinate_zdotc(
    const struct mtxmatrix_coordinate * x,
    const struct mtxmatrix_coordinate * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (!same_pattern(x, y)) return MTX_ERR_INCOMPATIBLE_PATTERN;
    if (x->symmetry == mtx_unsymmetric && y->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        struct mtxvector_array vecy;
        err = vectorise_array(&vecy, y);
        if (err) return err;
        return mtxvector_array_zdotc(&vecx, &vecy, dot, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_snrm2()’ computes the Frobenius norm of a
 * matrix in single precision floating point.
 */
int mtxmatrix_coordinate_snrm2(
    const struct mtxmatrix_coordinate * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        return mtxvector_array_snrm2(&vecx, nrm2, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_dnrm2()’ computes the Frobenius norm of a
 * matrix in double precision floating point.
 */
int mtxmatrix_coordinate_dnrm2(
    const struct mtxmatrix_coordinate * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        return mtxvector_array_dnrm2(&vecx, nrm2, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_coordinate_sasum(
    const struct mtxmatrix_coordinate * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        return mtxvector_array_sasum(&vecx, asum, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_coordinate_dasum(
    const struct mtxmatrix_coordinate * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->symmetry == mtx_unsymmetric) {
        struct mtxvector_array vecx;
        int err = vectorise_array(&vecx, x);
        if (err) return err;
        return mtxvector_array_dasum(&vecx, asum, num_flops);
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
    }
}

/**
 * ‘mtxmatrix_coordinate_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_coordinate_iamax(
    const struct mtxmatrix_coordinate * x,
    int * iamax)
{
    struct mtxvector_array vecx;
    int err = vectorise_array(&vecx, x);
    if (err) return err;
    return mtxvector_array_iamax(&vecx, iamax);
}

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
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    if (x->type != mtxvector_array || y->type != mtxvector_array)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_array * x_ = &x->storage.array;
    struct mtxvector_array * y_ = &y->storage.array;
    if (x_->field != A->field || y_->field != A->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != A->precision || y_->precision != A->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_sscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        const int * i = trans == mtx_notrans ? A->rowidx : A->colidx;
        const int * j = trans == mtx_notrans ? A->colidx : A->rowidx;
        if (A->field == mtx_field_real) {
            if (A->precision == mtx_single) {
                const float * Adata = A->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double * Adata = A->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (A->precision == mtx_single) {
                const int32_t * Adata = A->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const int64_t * Adata = A->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (A->field == mtx_field_real) {
            if (A->precision == mtx_single) {
                const float * Adata = A->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double * Adata = A->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (A->precision == mtx_single) {
                const int32_t * Adata = A->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const int64_t * Adata = A->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_dgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
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
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    if (x->type != mtxvector_array || y->type != mtxvector_array)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_array * x_ = &x->storage.array;
    struct mtxvector_array * y_ = &y->storage.array;
    if (x_->field != A->field || y_->field != A->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != A->precision || y_->precision != A->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }

    if (beta != 1) {
        err = mtxvector_dscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        const int * i = trans == mtx_notrans ? A->rowidx : A->colidx;
        const int * j = trans == mtx_notrans ? A->colidx : A->rowidx;
        if (A->field == mtx_field_real) {
            if (A->precision == mtx_single) {
                const float * Adata = A->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double * Adata = A->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int64_t k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (A->precision == mtx_single) {
                const int32_t * Adata = A->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const int64_t * Adata = A->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int64_t k = 0; k < A->size; k++)
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (A->field == mtx_field_real) {
            if (A->precision == mtx_single) {
                const float * Adata = A->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double * Adata = A->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
                for (int64_t k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (A->precision == mtx_single) {
                const int32_t * Adata = A->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const int64_t * Adata = A->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]] += alpha*Adata[k]*xdata[j[k]];
                    if (i[k] != j[k]) ydata[j[k]] += alpha*Adata[k]*xdata[i[k]];
                }
                if (num_flops) *num_flops += 3*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    for (int k = 0; k < A->size; k++) {
                        ydata[i[k]][0] += alpha*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                        ydata[i[k]][1] += alpha*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                        if (i[k] != j[k]) {
                            ydata[j[k]][0] += alpha*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                            ydata[j[k]][1] += alpha*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_nonzeros;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

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
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    if (x->type != mtxvector_array || y->type != mtxvector_array)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_array * x_ = &x->storage.array;
    struct mtxvector_array * y_ = &y->storage.array;
    if (x_->field != A->field || y_->field != A->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != A->precision || y_->precision != A->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (A->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_cscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (trans == mtx_notrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

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
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y,
    int64_t * num_flops)
{
    int err;
    if (x->type != mtxvector_array || y->type != mtxvector_array)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_array * x_ = &x->storage.array;
    struct mtxvector_array * y_ = &y->storage.array;
    if (x_->field != A->field || y_->field != A->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != A->precision || y_->precision != A->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (trans == mtx_notrans) {
        if (A->num_rows != y_->size || A->num_columns != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    } else if (trans == mtx_trans || trans == mtx_conjtrans) {
        if (A->num_columns != y_->size || A->num_rows != x_->size)
            return MTX_ERR_INCOMPATIBLE_SIZE;
    }
    if (A->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (beta[0] != 1 || beta[1] != 0) {
        err = mtxvector_zscal(beta, y, num_flops);
        if (err) return err;
    }

    if (A->symmetry == mtx_unsymmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (trans == mtx_notrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1])-alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0])+alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1])-alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                    ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0])+alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        const int * i = A->rowidx;
        const int * j = A->colidx;
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]+Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]-Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]-Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]+Adata[k][1]*xdata[i[k]][1]);
                    }
                }
                if (num_flops) *num_flops += 20*A->num_nonzeros;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                for (int k = 0; k < A->size; k++) {
                    ydata[i[k]][0] += alpha[0]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]) - alpha[1]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]);
                    ydata[i[k]][1] += alpha[0]*(Adata[k][0]*xdata[j[k]][1]-Adata[k][1]*xdata[j[k]][0]) + alpha[1]*(Adata[k][0]*xdata[j[k]][0]+Adata[k][1]*xdata[j[k]][1]);
                    if (i[k] != j[k]) {
                        ydata[j[k]][0] += alpha[0]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]) - alpha[1]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]);
                        ydata[j[k]][1] += alpha[0]*(Adata[k][0]*xdata[i[k]][1]+Adata[k][1]*xdata[i[k]][0]) + alpha[1]*(Adata[k][0]*xdata[i[k]][0]-Adata[k][1]*xdata[i[k]][1]);
                    }
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}
