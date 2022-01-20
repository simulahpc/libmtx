/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
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
 * Last modified: 2021-10-05
 *
 * Data structures for matrices in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/field.h>
#include <libmtx/matrix/matrix_array.h>
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
 * `mtxmatrix_array_free()' frees storage allocated for a matrix.
 */
void mtxmatrix_array_free(
    struct mtxmatrix_array * matrix)
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
}

/**
 * `mtxmatrix_array_alloc_copy()' allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_array_alloc_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/**
 * `mtxmatrix_array_init_copy()' allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_array_init_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src);

/*
 * Matrix array formats
 */

/**
 * `mtxmatrix_array_alloc()' allocates a matrix in array format.
 */
int mtxmatrix_array_alloc(
    struct mtxmatrix_array * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    int num_rows,
    int num_columns)
{
    int64_t size;
    if (__builtin_mul_overflow(num_rows, num_columns, &size)) {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }

    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            matrix->data.real_single =
                malloc(size * sizeof(*matrix->data.real_single));
            if (!matrix->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            matrix->data.real_double =
                malloc(size * sizeof(*matrix->data.real_double));
            if (!matrix->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            matrix->data.complex_single =
                malloc(size * sizeof(*matrix->data.complex_single));
            if (!matrix->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            matrix->data.complex_double =
                malloc(size * sizeof(*matrix->data.complex_double));
            if (!matrix->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            matrix->data.integer_single =
                malloc(size * sizeof(*matrix->data.integer_single));
            if (!matrix->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            matrix->data.integer_double =
                malloc(size * sizeof(*matrix->data.integer_double));
            if (!matrix->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    matrix->field = field;
    matrix->precision = precision;
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->size = size;
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_real_single()' allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_array_init_real_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_real, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_real_double()' allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_array_init_real_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_real, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_complex_single()' allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_array_init_complex_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const float (* data)[2])
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_complex, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++) {
        matrix->data.complex_single[k][0] = data[k][0];
        matrix->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_complex_double()' allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_array_init_complex_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const double (* data)[2])
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_complex, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++) {
        matrix->data.complex_double[k][0] = data[k][0];
        matrix->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_integer_single()' allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_array_init_integer_single(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int32_t * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_integer, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_init_integer_double()' allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_array_init_integer_double(
    struct mtxmatrix_array * matrix,
    int num_rows,
    int num_columns,
    const int64_t * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_integer, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/*
 * Row and column vectors
 */

/**
 * `mtxmatrix_array_alloc_row_vector()' allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmatrix_array_alloc_row_vector(
    const struct mtxmatrix_array * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type)
{
    if (vector_type == mtxvector_auto)
        vector_type = mtxvector_array;

    if (vector_type == mtxvector_array) {
        return mtxvector_alloc_array(
            vector, matrix->field, matrix->precision, matrix->num_columns);
    } else if (vector_type == mtxvector_coordinate) {
        return mtxvector_alloc_coordinate(
            vector, matrix->field, matrix->precision,
            matrix->num_columns, matrix->num_columns);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxmatrix_array_alloc_column_vector()' allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmatrix_array_alloc_column_vector(
    const struct mtxmatrix_array * matrix,
    struct mtxvector * vector,
    enum mtxvectortype vector_type)
{
    if (vector_type == mtxvector_auto)
        vector_type = mtxvector_array;

    if (vector_type == mtxvector_array) {
        return mtxvector_alloc_array(
            vector, matrix->field, matrix->precision, matrix->num_rows);
    } else if (vector_type == mtxvector_coordinate) {
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
 * `mtxmatrix_array_from_mtxfile()' converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_array_from_mtxfile(
    struct mtxmatrix_array * matrix,
    const struct mtxfile * mtxfile)
{
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from coordinate to array. */
    if (mtxfile->header.format != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_real_single(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_real_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_real_double(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_complex_single(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_complex_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_complex_double(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_integer_single(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_integer_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_integer_double(
                matrix, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        return MTX_ERR_INCOMPATIBLE_MTX_FIELD;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxmatrix_array_to_mtxfile()' converts a matrix to a matrix in
 * Matrix Market format.
 */
int mtxmatrix_array_to_mtxfile(
    const struct mtxmatrix_array * matrix,
    struct mtxfile * mtxfile)
{
    if (matrix->field == mtx_field_real) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_real_single(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.real_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_real_double(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_complex) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_complex_single(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.complex_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_complex_double(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_integer) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_integer_single(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.integer_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_integer_double(
                mtxfile, mtxfile_general, matrix->num_rows, matrix->num_columns,
                matrix->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_pattern) {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

/**
 * ‘mtxmatrix_array_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
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
int mtxmatrix_array_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    float beta,
    struct mtxvector * y)
{
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
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans : CblasTrans;
            cblas_sgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    float z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    float z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else if (A->precision == mtx_double) {
            const double * Adata = A->data.real_double;
            const double * xdata = x_->data.real_double;
            double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans : CblasTrans;
            cblas_dgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    double z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    double z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_complex) {
        if (A->precision == mtx_single) {
            const float (* Adata)[2] = A->data.complex_single;
            const float (* xdata)[2] = x_->data.complex_single;
            float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
                : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
            const float calpha[2] = {alpha, 0};
            const float cbeta[2] = {beta, 0};
            cblas_cgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                calpha, (const float *) Adata, A->num_columns,
                (const float *) xdata, 1, cbeta, (float *) ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    float z[2] = {0, 0};
                    for (int j = 0; j < A->num_columns; j++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                            Adata[i*A->num_columns+j][1]*xdata[j][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                            Adata[i*A->num_columns+j][1]*xdata[j][0];
                    }
                    ydata[i][0] = alpha*z[0] + beta*ydata[i][0];
                    ydata[i][1] = alpha*z[1] + beta*ydata[i][1];
                }
            } else if (trans == mtx_trans) {
                for (int j = 0; j < A->num_columns; j++) {
                    float z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else if (trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    float z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else if (A->precision == mtx_double) {
            const double (* Adata)[2] = A->data.complex_double;
            const double (* xdata)[2] = x_->data.complex_double;
            double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
                : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
            const double zalpha[2] = {alpha, 0};
            const double zbeta[2] = {beta, 0};
            cblas_zgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                zalpha, (const double *) Adata, A->num_columns,
                (const double *) xdata, 1, zbeta, (double *) ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    double z[2] = {0, 0};
                    for (int j = 0; j < A->num_columns; j++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                            Adata[i*A->num_columns+j][1]*xdata[j][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                            Adata[i*A->num_columns+j][1]*xdata[j][0];
                    }
                    ydata[i][0] = alpha*z[0] + beta*ydata[i][0];
                    ydata[i][1] = alpha*z[1] + beta*ydata[i][1];
                }
            } else if (trans == mtx_trans) {
                for (int j = 0; j < A->num_columns; j++) {
                    double z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else if (trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    double z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_integer) {
        if (A->precision == mtx_single) {
            const int32_t * Adata = A->data.integer_single;
            const int32_t * xdata = x_->data.integer_single;
            int32_t * ydata = y_->data.integer_single;
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    int32_t z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    int32_t z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const int64_t * Adata = A->data.integer_double;
            const int64_t * xdata = x_->data.integer_double;
            int64_t * ydata = y_->data.integer_double;
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    int64_t z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    int64_t z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
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
 * ‘mtxmatrix_array_dgemv()’ multiplies a matrix ‘A’ or its transpose
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
int mtxmatrix_array_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y)
{
    if (x->type != mtxvector_array || y->type != mtxvector_array)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;

    const struct mtxvector_array * x_ = &x->storage.array;
    struct mtxvector_array * y_ = &y->storage.array;
    if (A->num_rows != y_->size ||
        A->num_columns != x_->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x_->field != A->field || y_->field != A->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x_->precision != A->precision || y_->precision != A->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;

    if (A->field == mtx_field_real) {
        if (A->precision == mtx_single) {
            const float * Adata = A->data.real_single;
            const float * xdata = x_->data.real_single;
            float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans : CblasTrans;
            cblas_sgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    float z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    float z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else if (A->precision == mtx_double) {
            const double * Adata = A->data.real_double;
            const double * xdata = x_->data.real_double;
            double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans : CblasTrans;
            cblas_dgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    double z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    double z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_complex) {
        if (A->precision == mtx_single) {
            const float (* Adata)[2] = A->data.complex_single;
            const float (* xdata)[2] = x_->data.complex_single;
            float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
                : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
            const float calpha[2] = {alpha, 0};
            const float cbeta[2] = {beta, 0};
            cblas_cgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                calpha, (const float *) Adata, A->num_columns,
                (const float *) xdata, 1, cbeta, (float *) ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    float z[2] = {0, 0};
                    for (int j = 0; j < A->num_columns; j++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                            Adata[i*A->num_columns+j][1]*xdata[j][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                            Adata[i*A->num_columns+j][1]*xdata[j][0];
                    }
                    ydata[i][0] = alpha*z[0] + beta*ydata[i][0];
                    ydata[i][1] = alpha*z[1] + beta*ydata[i][1];
                }
            } else if (trans == mtx_trans) {
                for (int j = 0; j < A->num_columns; j++) {
                    float z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else if (trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    float z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else if (A->precision == mtx_double) {
            const double (* Adata)[2] = A->data.complex_double;
            const double (* xdata)[2] = x_->data.complex_double;
            double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
                : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
            const double zalpha[2] = {alpha, 0};
            const double zbeta[2] = {beta, 0};
            cblas_zgemv(
                CblasRowMajor, transA, A->num_rows, A->num_columns,
                zalpha, (const double *) Adata, A->num_columns,
                (const double *) xdata, 1, zbeta, (double *) ydata, 1);
#else
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    double z[2] = {0, 0};
                    for (int j = 0; j < A->num_columns; j++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                            Adata[i*A->num_columns+j][1]*xdata[j][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                            Adata[i*A->num_columns+j][1]*xdata[j][0];
                    }
                    ydata[i][0] = alpha*z[0] + beta*ydata[i][0];
                    ydata[i][1] = alpha*z[1] + beta*ydata[i][1];
                }
            } else if (trans == mtx_trans) {
                for (int j = 0; j < A->num_columns; j++) {
                    double z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else if (trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    double z[2] = {0, 0};
                    for (int i = 0; i < A->num_rows; i++) {
                        z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                            Adata[i*A->num_columns+j][1]*xdata[i][1];
                        z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                            Adata[i*A->num_columns+j][1]*xdata[i][0];
                    }
                    ydata[j][0] = alpha*z[0] + beta*ydata[j][0];
                    ydata[j][1] = alpha*z[1] + beta*ydata[j][1];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (A->field == mtx_field_integer) {
        if (A->precision == mtx_single) {
            const int32_t * Adata = A->data.integer_single;
            const int32_t * xdata = x_->data.integer_single;
            int32_t * ydata = y_->data.integer_single;
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    int32_t z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    int32_t z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
                }
            } else {
                return MTX_ERR_INVALID_TRANS_TYPE;
            }
        } else if (A->precision == mtx_double) {
            const int64_t * Adata = A->data.integer_double;
            const int64_t * xdata = x_->data.integer_double;
            int64_t * ydata = y_->data.integer_double;
            if (trans == mtx_notrans) {
                for (int i = 0; i < A->num_rows; i++) {
                    int64_t z = 0;
                    for (int j = 0; j < A->num_columns; j++)
                        z += Adata[i*A->num_columns+j]*xdata[j];
                    ydata[i] = alpha*z + beta*ydata[i];
                }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                for (int j = 0; j < A->num_columns; j++) {
                    int64_t z = 0;
                    for (int i = 0; i < A->num_rows; i++)
                        z += Adata[i*A->num_columns+j]*xdata[i];
                    ydata[j] = alpha*z + beta*ydata[j];
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
 * ‘mtxmatrix_array_cgemv()’ multiplies a complex-valued matrix ‘A’,
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
int mtxmatrix_array_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    float beta[2],
    struct mtxvector * y)
{
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
#ifdef LIBMTX_HAVE_BLAS
        enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
            : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
        cblas_cgemv(
            CblasRowMajor, transA, A->num_rows, A->num_columns,
            alpha, (const float *) Adata, A->num_columns,
            (const float *) xdata, 1, beta, (float *) ydata, 1);
#else
        if (trans == mtx_notrans) {
            for (int i = 0; i < A->num_rows; i++) {
                float z[2] = {0, 0};
                for (int j = 0; j < A->num_columns; j++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                        Adata[i*A->num_columns+j][1]*xdata[j][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                        Adata[i*A->num_columns+j][1]*xdata[j][0];
                }
                float yold[2] = {ydata[i][0], ydata[i][1]};
                ydata[i][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[i][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_trans) {
            for (int j = 0; j < A->num_columns; j++) {
                float z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                float yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_conjtrans) {
            for (int j = 0; j < A->num_columns; j++) {
                float z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                float yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
#endif
    } else if (A->precision == mtx_double) {
        const double (* Adata)[2] = A->data.complex_double;
        const double (* xdata)[2] = x_->data.complex_double;
        double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
        enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
            : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
        const double zalpha[2] = {alpha[0], alpha[1]};
        const double zbeta[2] = {beta[0], beta[1]};
        cblas_zgemv(
            CblasRowMajor, transA, A->num_rows, A->num_columns,
            zalpha, (const double *) Adata, A->num_columns,
            (const double *) xdata, 1, zbeta, (double *) ydata, 1);
#else
        if (trans == mtx_notrans) {
            for (int i = 0; i < A->num_rows; i++) {
                double z[2] = {0, 0};
                for (int j = 0; j < A->num_columns; j++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                        Adata[i*A->num_columns+j][1]*xdata[j][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                        Adata[i*A->num_columns+j][1]*xdata[j][0];
                }
                double yold[2] = {ydata[i][0], ydata[i][1]};
                ydata[i][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[i][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_trans) {
            for (int j = 0; j < A->num_columns; j++) {
                double z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                double yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_conjtrans) {
            for (int j = 0; j < A->num_columns; j++) {
                double z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                double yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
#endif
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
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
int mtxmatrix_array_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmatrix_array * A,
    const struct mtxvector * x,
    double beta[2],
    struct mtxvector * y)
{
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
#ifdef LIBMTX_HAVE_BLAS
        enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
            : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
        const float calpha[2] = {alpha[0], alpha[1]};
        const float cbeta[2] = {beta[0], beta[1]};
        cblas_cgemv(
            CblasRowMajor, transA, A->num_rows, A->num_columns,
            calpha, (const float *) Adata, A->num_columns,
            (const float *) xdata, 1, cbeta, (float *) ydata, 1);
#else
        if (trans == mtx_notrans) {
            for (int i = 0; i < A->num_rows; i++) {
                float z[2] = {0, 0};
                for (int j = 0; j < A->num_columns; j++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                        Adata[i*A->num_columns+j][1]*xdata[j][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                        Adata[i*A->num_columns+j][1]*xdata[j][0];
                }
                float yold[2] = {ydata[i][0], ydata[i][1]};
                ydata[i][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[i][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_trans) {
            for (int j = 0; j < A->num_columns; j++) {
                float z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                float yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_conjtrans) {
            for (int j = 0; j < A->num_columns; j++) {
                float z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                float yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
#endif
    } else if (A->precision == mtx_double) {
        const double (* Adata)[2] = A->data.complex_double;
        const double (* xdata)[2] = x_->data.complex_double;
        double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
        enum CBLAS_TRANSPOSE transA = (trans == mtx_notrans) ? CblasNoTrans
            : ((trans == mtx_trans) ? CblasTrans : CblasConjTrans);
        cblas_zgemv(
            CblasRowMajor, transA, A->num_rows, A->num_columns,
            alpha, (const double *) Adata, A->num_columns,
            (const double *) xdata, 1, beta, (double *) ydata, 1);
#else
        if (trans == mtx_notrans) {
            for (int i = 0; i < A->num_rows; i++) {
                double z[2] = {0, 0};
                for (int j = 0; j < A->num_columns; j++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[j][0] -
                        Adata[i*A->num_columns+j][1]*xdata[j][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[j][1] +
                        Adata[i*A->num_columns+j][1]*xdata[j][0];
                }
                double yold[2] = {ydata[i][0], ydata[i][1]};
                ydata[i][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[i][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_trans) {
            for (int j = 0; j < A->num_columns; j++) {
                double z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] -
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] +
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                double yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else if (trans == mtx_conjtrans) {
            for (int j = 0; j < A->num_columns; j++) {
                double z[2] = {0, 0};
                for (int i = 0; i < A->num_rows; i++) {
                    z[0] += Adata[i*A->num_columns+j][0]*xdata[i][0] +
                        Adata[i*A->num_columns+j][1]*xdata[i][1];
                    z[1] += Adata[i*A->num_columns+j][0]*xdata[i][1] -
                        Adata[i*A->num_columns+j][1]*xdata[i][0];
                }
                double yold[2] = {ydata[j][0], ydata[j][1]};
                ydata[j][0] = alpha[0]*z[0] - alpha[1]*z[1]
                    + beta[0]*yold[0] - beta[1]*yold[1];
                ydata[j][1] = alpha[0]*z[1] + alpha[1]*z[0]
                    + beta[0]*yold[1] + beta[1]*yold[0];
            }
        } else {
            return MTX_ERR_INVALID_TRANS_TYPE;
        }
#endif
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}
