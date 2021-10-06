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
 * Data structures for matrices in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/field.h>
#include <libmtx/matrix/matrix_coordinate.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

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
 * `mtxmatrix_coordinate_free()' frees storage allocated for a matrix.
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
 * `mtxmatrix_coordinate_alloc_copy()' allocates a copy of a matrix
 * without initialising the values.
 */
int mtxmatrix_coordinate_alloc_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src);

/**
 * `mtxmatrix_coordinate_init_copy()' allocates a copy of a matrix and
 * also copies the values.
 */
int mtxmatrix_coordinate_init_copy(
    struct mtxmatrix_coordinate * dst,
    const struct mtxmatrix_coordinate * src);

/*
 * Matrix coordinate formats
 */

/**
 * `mtxmatrix_coordinate_alloc()' allocates a matrix in coordinate
 * format.
 */
int mtxmatrix_coordinate_alloc(
    struct mtxmatrix_coordinate * matrix,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros)
{
    matrix->rowidx = malloc(num_nonzeros * sizeof(int));
    if (!matrix->rowidx)
        return MTX_ERR_ERRNO;
    matrix->colidx = malloc(num_nonzeros * sizeof(int));
    if (!matrix->colidx) {
        free(matrix->rowidx);
        return MTX_ERR_ERRNO;
    }
    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            matrix->data.real_single =
                malloc(num_nonzeros * sizeof(*matrix->data.real_single));
            if (!matrix->data.real_single) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            matrix->data.real_double =
                malloc(num_nonzeros * sizeof(*matrix->data.real_double));
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
                malloc(num_nonzeros * sizeof(*matrix->data.complex_single));
            if (!matrix->data.complex_single) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            matrix->data.complex_double =
                malloc(num_nonzeros * sizeof(*matrix->data.complex_double));
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
                malloc(num_nonzeros * sizeof(*matrix->data.integer_single));
            if (!matrix->data.integer_single) {
                free(matrix->colidx);
                free(matrix->rowidx);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            matrix->data.integer_double =
                malloc(num_nonzeros * sizeof(*matrix->data.integer_double));
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
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->num_nonzeros = num_nonzeros;
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_real_single()' allocates and initialises
 * a matrix in coordinate format with real, single precision
 * coefficients.
 */
int mtxmatrix_coordinate_init_real_single(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_real, mtx_single, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.real_single[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_real_double()' allocates and initialises
 * a matrix in coordinate format with real, double precision
 * coefficients.
 */
int mtxmatrix_coordinate_init_real_double(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_real, mtx_double, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.real_double[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_complex_single()' allocates and
 * initialises a matrix in coordinate format with complex, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_complex_single(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_complex, mtx_single, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.complex_single[k][0] = data[k][0];
        matrix->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_complex_double()' allocates and
 * initialises a matrix in coordinate format with complex, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_complex_double(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2])
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_complex, mtx_double, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.complex_double[k][0] = data[k][0];
        matrix->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_integer_single()' allocates and
 * initialises a matrix in coordinate format with integer, single
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_integer_single(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_integer, mtx_single, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.integer_single[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_integer_double()' allocates and
 * initialises a matrix in coordinate format with integer, double
 * precision coefficients.
 */
int mtxmatrix_coordinate_init_integer_double(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_integer, mtx_double, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
        matrix->data.integer_double[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_init_pattern()' allocates and initialises a
 * matrix in coordinate format with boolean coefficients.
 */
int mtxmatrix_coordinate_init_pattern(
    struct mtxmatrix_coordinate * matrix,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx)
{
    int err = mtxmatrix_coordinate_alloc(
        matrix, mtx_field_pattern, mtx_single, num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        matrix->rowidx[k] = rowidx[k];
        matrix->colidx[k] = colidx[k];
    }
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxmatrix_coordinate_from_mtxfile()' converts a matrix in Matrix
 * Market format to a matrix.
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

    /* TODO: If needed, we could convert from a symmetric
     * representation. */
    if (mtxfile->header.symmetry != mtxfile_general)
        return MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY;

    int num_rows = mtxfile->size.num_rows;
    int num_columns = mtxfile->size.num_columns;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;

    if (mtxfile->header.field == mtxfile_real) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_real, mtxfile->precision,
            num_rows, num_columns, num_nonzeros);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                matrix->rowidx[k] = data[k].i;
                matrix->colidx[k] = data[k].j;
                matrix->data.real_single[k] = data[k].a;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                matrix->rowidx[k] = data[k].i;
                matrix->colidx[k] = data[k].j;
                matrix->data.real_double[k] = data[k].a;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_complex, mtxfile->precision,
            num_rows, num_columns, num_nonzeros);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                matrix->rowidx[k] = data[k].i;
                matrix->colidx[k] = data[k].j;
                matrix->data.complex_single[k][0] = data[k].a[0];
                matrix->data.complex_single[k][1] = data[k].a[1];
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                matrix->rowidx[k] = data[k].i;
                matrix->colidx[k] = data[k].j;
                matrix->data.complex_double[k][0] = data[k].a[0];
                matrix->data.complex_double[k][1] = data[k].a[1];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_integer, mtxfile->precision,
            num_rows, num_columns, num_nonzeros);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                matrix->rowidx[k] = data[k].i;
                matrix->colidx[k] = data[k].j;
                matrix->data.integer_single[k] = data[k].a;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                matrix->rowidx[k] = data[k].i;
                matrix->colidx[k] = data[k].j;
                matrix->data.integer_double[k] = data[k].a;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        err = mtxmatrix_coordinate_alloc(
            matrix, mtx_field_pattern, mtx_single,
            num_rows, num_columns, num_nonzeros);
        if (err)
            return err;
        const struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < num_nonzeros; k++) {
            matrix->rowidx[k] = data[k].i;
            matrix->colidx[k] = data[k].j;
        }
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_coordinate_to_mtxfile()' converts a matrix to a matrix
 * in Matrix Market format.
 */
int mtxmatrix_coordinate_to_mtxfile(
    const struct mtxmatrix_coordinate * matrix,
    struct mtxfile * mtxfile)
{
    int err;
    if (matrix->field == mtx_field_real) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_real, mtxfile_general, matrix->precision,
            matrix->num_rows, matrix->num_columns, matrix->num_nonzeros);
        if (err)
            return err;
        if (matrix->precision == mtx_single) {
            struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
                data[k].i = matrix->rowidx[k];
                data[k].j = matrix->colidx[k];
                data[k].a = matrix->data.real_single[k];
            }
        } else if (matrix->precision == mtx_double) {
            struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
                data[k].i = matrix->rowidx[k];
                data[k].j = matrix->colidx[k];
                data[k].a = matrix->data.real_double[k];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_complex) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_complex, mtxfile_general, matrix->precision,
            matrix->num_rows, matrix->num_columns, matrix->num_nonzeros);
        if (err)
            return err;
        if (matrix->precision == mtx_single) {
            struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
                data[k].i = matrix->rowidx[k];
                data[k].j = matrix->colidx[k];
                data[k].a[0] = matrix->data.complex_single[k][0];
                data[k].a[1] = matrix->data.complex_single[k][1];
            }
        } else if (matrix->precision == mtx_double) {
            struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
                data[k].i = matrix->rowidx[k];
                data[k].j = matrix->colidx[k];
                data[k].a[0] = matrix->data.complex_double[k][0];
                data[k].a[1] = matrix->data.complex_double[k][1];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_integer) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_integer, mtxfile_general, matrix->precision,
            matrix->num_rows, matrix->num_columns, matrix->num_nonzeros);
        if (err)
            return err;
        if (matrix->precision == mtx_single) {
            struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
                data[k].i = matrix->rowidx[k];
                data[k].j = matrix->colidx[k];
                data[k].a = matrix->data.integer_single[k];
            }
        } else if (matrix->precision == mtx_double) {
            struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
                data[k].i = matrix->rowidx[k];
                data[k].j = matrix->colidx[k];
                data[k].a = matrix->data.integer_double[k];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_pattern) {
        err = mtxfile_alloc_matrix_coordinate(
            mtxfile, mtxfile_pattern, mtxfile_general, mtx_single,
            matrix->num_rows, matrix->num_columns, matrix->num_nonzeros);
        if (err)
            return err;
        struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < matrix->num_nonzeros; k++) {
            data[k].i = matrix->rowidx[k];
            data[k].j = matrix->colidx[k];
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}
