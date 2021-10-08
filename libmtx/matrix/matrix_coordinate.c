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
#include <libmtx/vector/vector.h>
#include <libmtx/vector/vector_array.h>

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
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_sgemv(
    enum mtx_trans_type trans,
    float alpha,
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y)
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

    if (A->field == mtx_field_real) {
        if (A->precision == mtx_single) {
            const float * Adata = A->data.real_single;
            const float * xdata = x_->data.real_single;
            float * ydata = y_->data.real_single;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float * zdata = z.data.real_single;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float * zdata = z.data.real_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const double * Adata = A->data.real_double;
            const double * xdata = x_->data.real_double;
            double * ydata = y_->data.real_double;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double * zdata = z.data.real_double;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double * zdata = z.data.real_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_complex) {
        if (A->precision == mtx_single) {
            const float (* Adata)[2] = A->data.complex_single;
            const float (* xdata)[2] = x_->data.complex_single;
            float (* ydata)[2] = y_->data.complex_single;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i][0] = ydata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float (* zdata)[2] = z.data.complex_single;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i][0] = zdata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i][0] += (Adata[k][0]*xdata[j][0] -
                                        Adata[k][1]*xdata[j][1]);
                        zdata[i][1] += (Adata[k][0]*xdata[j][1] +
                                        Adata[k][1]*xdata[j][0]);

                    }
                    for (int i = 0; i < A->num_rows; i++) {
                        ydata[i][0] = alpha*zdata[i][0] + beta*ydata[i][0];
                        ydata[i][1] = alpha*zdata[i][1] + beta*ydata[i][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float (* zdata)[2] = z.data.complex_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float (* zdata)[2] = z.data.complex_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const double (* Adata)[2] = A->data.complex_double;
            const double (* xdata)[2] = x_->data.complex_double;
            double (* ydata)[2] = y_->data.complex_double;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i][0] = ydata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double (* zdata)[2] = z.data.complex_double;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i][0] = zdata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i][0] += (Adata[k][0]*xdata[j][0] -
                                        Adata[k][1]*xdata[j][1]);
                        zdata[i][1] += (Adata[k][0]*xdata[j][1] +
                                        Adata[k][1]*xdata[j][0]);

                    }
                    for (int i = 0; i < A->num_rows; i++) {
                        ydata[i][0] = alpha*zdata[i][0] + beta*ydata[i][0];
                        ydata[i][1] = alpha*zdata[i][1] + beta*ydata[i][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double (* zdata)[2] = z.data.complex_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double (* zdata)[2] = z.data.complex_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_integer) {
        if (A->precision == mtx_single) {
            const int32_t * Adata = A->data.integer_single;
            const int32_t * xdata = x_->data.integer_single;
            int32_t * ydata = y_->data.integer_single;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int32_t * zdata = z.data.integer_single;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int32_t * zdata = z.data.integer_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const int64_t * Adata = A->data.integer_double;
            const int64_t * xdata = x_->data.integer_double;
            int64_t * ydata = y_->data.integer_double;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int64_t * zdata = z.data.integer_double;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int64_t * zdata = z.data.integer_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
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
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the
 * size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_dgemv(
    enum mtx_trans_type trans,
    double alpha,
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y)
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

    if (A->field == mtx_field_real) {
        if (A->precision == mtx_single) {
            const float * Adata = A->data.real_single;
            const float * xdata = x_->data.real_single;
            float * ydata = y_->data.real_single;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float * zdata = z.data.real_single;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float * zdata = z.data.real_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const double * Adata = A->data.real_double;
            const double * xdata = x_->data.real_double;
            double * ydata = y_->data.real_double;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double * zdata = z.data.real_double;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double * zdata = z.data.real_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_complex) {
        if (A->precision == mtx_single) {
            const float (* Adata)[2] = A->data.complex_single;
            const float (* xdata)[2] = x_->data.complex_single;
            float (* ydata)[2] = y_->data.complex_single;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i][0] = ydata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float (* zdata)[2] = z.data.complex_single;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i][0] = zdata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i][0] += (Adata[k][0]*xdata[j][0] -
                                        Adata[k][1]*xdata[j][1]);
                        zdata[i][1] += (Adata[k][0]*xdata[j][1] +
                                        Adata[k][1]*xdata[j][0]);

                    }
                    for (int i = 0; i < A->num_rows; i++) {
                        ydata[i][0] = alpha*zdata[i][0] + beta*ydata[i][0];
                        ydata[i][1] = alpha*zdata[i][1] + beta*ydata[i][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float (* zdata)[2] = z.data.complex_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    float (* zdata)[2] = z.data.complex_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const double (* Adata)[2] = A->data.complex_double;
            const double (* xdata)[2] = x_->data.complex_double;
            double (* ydata)[2] = y_->data.complex_double;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i][0] = ydata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i][0] += alpha * (Adata[k][0]*xdata[j][0] -
                                                Adata[k][1]*xdata[j][1]);
                        ydata[i][1] += alpha * (Adata[k][0]*xdata[j][1] +
                                                Adata[k][1]*xdata[j][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double (* zdata)[2] = z.data.complex_double;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i][0] = zdata[i][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i][0] += (Adata[k][0]*xdata[j][0] -
                                        Adata[k][1]*xdata[j][1]);
                        zdata[i][1] += (Adata[k][0]*xdata[j][1] +
                                        Adata[k][1]*xdata[j][0]);

                    }
                    for (int i = 0; i < A->num_rows; i++) {
                        ydata[i][0] = alpha*zdata[i][0] + beta*ydata[i][0];
                        ydata[i][1] = alpha*zdata[i][1] + beta*ydata[i][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] -
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] +
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double (* zdata)[2] = z.data.complex_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j][0] = ydata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j][0] += alpha * (Adata[k][0]*xdata[i][0] +
                                                Adata[k][1]*xdata[i][1]);
                        ydata[j][1] += alpha * (Adata[k][0]*xdata[i][1] -
                                                Adata[k][1]*xdata[i][0]);
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    double (* zdata)[2] = z.data.complex_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j][0] = zdata[j][1] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                        zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                    }
                    for (int j = 0; j < A->num_columns; j++) {
                        ydata[j][0] = alpha*zdata[j][0] + beta*ydata[j][0];
                        ydata[j][1] = alpha*zdata[j][1] + beta*ydata[j][1];
                    }
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_integer) {
        if (A->precision == mtx_single) {
            const int32_t * Adata = A->data.integer_single;
            const int32_t * xdata = x_->data.integer_single;
            int32_t * ydata = y_->data.integer_single;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int32_t * zdata = z.data.integer_single;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int32_t * zdata = z.data.integer_single;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const int64_t * Adata = A->data.integer_double;
            const int64_t * xdata = x_->data.integer_double;
            int64_t * ydata = y_->data.integer_double;
            if (trans == mtx_notrans) {
                if (beta == 0) {
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int64_t * zdata = z.data.integer_double;
                    for (int i = 0; i < A->num_rows; i++)
                        zdata[i] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[i] += Adata[k]*xdata[j];
                    }
                    for (int i = 0; i < A->num_rows; i++)
                        ydata[i] = alpha*zdata[i] + beta*ydata[i];
                    mtxvector_array_free(&z);
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (beta == 0) {
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else if (beta == 1) {
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                } else {
                    struct mtxvector_array z;
                    err = mtxvector_array_alloc_copy(&z, y_);
                    if (err)
                        return err;
                    int64_t * zdata = z.data.integer_double;
                    for (int j = 0; j < A->num_columns; j++)
                        zdata[j] = 0;
                    for (int64_t k = 0; k < A->num_nonzeros; k++) {
                        int i = A->rowidx[k];
                        int j = A->colidx[k];
                        zdata[j] += Adata[k]*xdata[i];
                    }
                    for (int j = 0; j < A->num_columns; j++)
                        ydata[j] = alpha*zdata[j] + beta*ydata[j];
                    mtxvector_array_free(&z);
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_coordinate_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be of ‘array’ type, and they must have
 * the same field and precision as the matrix ‘A’. Moreover, if
 * ‘trans’ is ‘mtx_notrans’, then the size of ‘x’ must equal the
 * number of columns of ‘A’ and the size of ‘y’ must equal the number
 * of rows of ‘A’. if ‘trans’ is ‘mtx_trans’ or ‘mtx_conjtrans’, then
 * the size of ‘x’ must equal the number of rows of ‘A’ and the
 * size of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_coordinate_cgemv(
    enum mtx_trans_type trans,
    float alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y)
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

    if (A->precision == mtx_single) {
        const float (* Adata)[2] = A->data.complex_single;
        const float (* xdata)[2] = x_->data.complex_single;
        float (* ydata)[2] = y_->data.complex_single;
        if (trans == mtx_notrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int i = 0; i < A->num_rows; i++)
                    ydata[i][0] = ydata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                  Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                  Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                float (* zdata)[2] = z.data.complex_single;
                for (int i = 0; i < A->num_rows; i++)
                    zdata[i][0] = zdata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[i][0] += Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1];
                    zdata[i][1] += Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0];
                }
                for (int i = 0; i < A->num_rows; i++) {
                    float w[2] = {ydata[i][0], ydata[i][1]};
                    ydata[i][0] = alpha[0]*zdata[i][0]-alpha[1]*zdata[i][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[i][1] = alpha[0]*zdata[i][1]+alpha[1]*zdata[i][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_trans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                float (* zdata)[2] = z.data.complex_single;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    float w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_conjtrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                float (* zdata)[2] = z.data.complex_single;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    float w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
    } else if (A->precision == mtx_double) {
        const double (* Adata)[2] = A->data.complex_double;
        const double (* xdata)[2] = x_->data.complex_double;
        double (* ydata)[2] = y_->data.complex_double;
        if (trans == mtx_notrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int i = 0; i < A->num_rows; i++)
                    ydata[i][0] = ydata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                   Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                   Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                double (* zdata)[2] = z.data.complex_double;
                for (int i = 0; i < A->num_rows; i++)
                    zdata[i][0] = zdata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[i][0] += (Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    zdata[i][1] += (Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                }
                for (int i = 0; i < A->num_rows; i++) {
                    double w[2] = {ydata[i][0], ydata[i][1]};
                    ydata[i][0] = alpha[0]*zdata[i][0]-alpha[1]*zdata[i][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[i][1] = alpha[0]*zdata[i][1]+alpha[1]*zdata[i][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_trans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                double (* zdata)[2] = z.data.complex_double;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    double w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_conjtrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                double (* zdata)[2] = z.data.complex_double;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    double w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
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
    enum mtx_trans_type trans,
    double alpha[2],
    const struct mtxmatrix_coordinate * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y)
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

    if (A->precision == mtx_single) {
        const float (* Adata)[2] = A->data.complex_single;
        const float (* xdata)[2] = x_->data.complex_single;
        float (* ydata)[2] = y_->data.complex_single;
        if (trans == mtx_notrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int i = 0; i < A->num_rows; i++)
                    ydata[i][0] = ydata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                  Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                  Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                float (* zdata)[2] = z.data.complex_single;
                for (int i = 0; i < A->num_rows; i++)
                    zdata[i][0] = zdata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[i][0] += Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1];
                    zdata[i][1] += Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0];
                }
                for (int i = 0; i < A->num_rows; i++) {
                    float w[2] = {ydata[i][0], ydata[i][1]};
                    ydata[i][0] = alpha[0]*zdata[i][0]-alpha[1]*zdata[i][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[i][1] = alpha[0]*zdata[i][1]+alpha[1]*zdata[i][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_trans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                float (* zdata)[2] = z.data.complex_single;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    float w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_conjtrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    float z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                  Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                float (* zdata)[2] = z.data.complex_single;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    float w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
    } else if (A->precision == mtx_double) {
        const double (* Adata)[2] = A->data.complex_double;
        const double (* xdata)[2] = x_->data.complex_double;
        double (* ydata)[2] = y_->data.complex_double;
        if (trans == mtx_notrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int i = 0; i < A->num_rows; i++)
                    ydata[i][0] = ydata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                   Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[j][0] - Adata[k][1]*xdata[j][1],
                                   Adata[k][0]*xdata[j][1] + Adata[k][1]*xdata[j][0]};
                    ydata[i][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[i][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                double (* zdata)[2] = z.data.complex_double;
                for (int i = 0; i < A->num_rows; i++)
                    zdata[i][0] = zdata[i][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[i][0] += (Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    zdata[i][1] += (Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                }
                for (int i = 0; i < A->num_rows; i++) {
                    double w[2] = {ydata[i][0], ydata[i][1]};
                    ydata[i][0] = alpha[0]*zdata[i][0]-alpha[1]*zdata[i][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[i][1] = alpha[0]*zdata[i][1]+alpha[1]*zdata[i][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_trans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0] - Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1] + Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                double (* zdata)[2] = z.data.complex_double;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    double w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else if (trans == mtx_conjtrans) {
            if (beta[0] == 0 && beta[1] == 0) {
                for (int j = 0; j < A->num_columns; j++)
                    ydata[j][0] = ydata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else if (beta[0] == 1 && beta[1] == 0) {
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    double z[2] = {Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1],
                                   Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]};
                    ydata[j][0] += alpha[0]*z[0]-alpha[1]*z[1];
                    ydata[j][1] += alpha[0]*z[1]+alpha[1]*z[0];
                }
            } else {
                struct mtxvector_array z;
                err = mtxvector_array_alloc_copy(&z, y_);
                if (err)
                    return err;
                double (* zdata)[2] = z.data.complex_double;
                for (int j = 0; j < A->num_columns; j++)
                    zdata[j][0] = zdata[j][1] = 0;
                for (int64_t k = 0; k < A->num_nonzeros; k++) {
                    int i = A->rowidx[k];
                    int j = A->colidx[k];
                    zdata[j][0] += Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1];
                    zdata[j][1] += Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0];
                }
                for (int j = 0; j < A->num_columns; j++) {
                    double w[2] = {ydata[j][0], ydata[j][1]};
                    ydata[j][0] = alpha[0]*zdata[j][0]-alpha[1]*zdata[j][1]
                        + beta[0]*w[0]-beta[1]*w[1];
                    ydata[j][1] = alpha[0]*zdata[j][1]+alpha[1]*zdata[j][0]
                        + beta[0]*w[1]+beta[1]*w[0];
                }
                mtxvector_array_free(&z);
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}
