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
 * Last modified: 2022-03-19
 *
 * Data structures for matrices in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/matrix_array.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/symmetry.h>
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
 * ‘mtxmatrix_array_free()’ frees storage allocated for a matrix.
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
 * ‘mtxmatrix_array_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmatrix_array_alloc_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src)
{
    return mtxmatrix_array_alloc(
        dst, src->field, src->precision, src->symmetry,
        src->num_rows, src->num_columns);
}

/**
 * ‘mtxmatrix_array_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmatrix_array_init_copy(
    struct mtxmatrix_array * dst,
    const struct mtxmatrix_array * src)
{
    int err = mtxmatrix_array_alloc_copy(dst, src);
    if (err) return err;
    err = mtxmatrix_array_copy(dst, src);
    if (err) return err;
    return MTX_SUCCESS;
}

/*
 * Matrix array formats
 */

/**
 * ‘mtxmatrix_array_alloc()’ allocates a matrix in array format.
 */
int mtxmatrix_array_alloc(
    struct mtxmatrix_array * matrix,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns)
{
    int64_t num_entries;
    if (__builtin_mul_overflow(num_rows, num_columns, &num_entries)) {
        errno = EOVERFLOW;
        return MTX_ERR_ERRNO;
    }

    int64_t size;
    if (symmetry == mtx_unsymmetric) {
        size = num_entries;
    } else if (num_rows == num_columns &&
               (symmetry == mtx_symmetric ||
                (symmetry == mtx_hermitian && field == mtx_field_complex)))
    {
        size = (num_entries+num_rows) / 2;
    } else if (num_rows == num_columns && symmetry == mtx_skew_symmetric) {
        size = (num_entries-num_rows) / 2;
    } else {
        return MTX_ERR_INVALID_SYMMETRY;
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
    matrix->symmetry = symmetry;
    matrix->num_rows = num_rows;
    matrix->num_columns = num_columns;
    matrix->num_entries = num_entries;
    matrix->size = size;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_init_real_single()’ allocates and initialises a
 * matrix in array format with real, single precision coefficients.
 */
int mtxmatrix_array_init_real_single(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_real, mtx_single, symmetry, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_init_real_double()’ allocates and initialises a
 * matrix in array format with real, double precision coefficients.
 */
int mtxmatrix_array_init_real_double(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_real, mtx_double, symmetry, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_init_complex_single()’ allocates and initialises a
 * matrix in array format with complex, single precision coefficients.
 */
int mtxmatrix_array_init_complex_single(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2])
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_complex, mtx_single, symmetry, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++) {
        matrix->data.complex_single[k][0] = data[k][0];
        matrix->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_init_complex_double()’ allocates and initialises a
 * matrix in array format with complex, double precision coefficients.
 */
int mtxmatrix_array_init_complex_double(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2])
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_complex, mtx_double, symmetry, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++) {
        matrix->data.complex_double[k][0] = data[k][0];
        matrix->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_init_integer_single()’ allocates and initialises a
 * matrix in array format with integer, single precision coefficients.
 */
int mtxmatrix_array_init_integer_single(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_integer, mtx_single, symmetry, num_rows, num_columns);
    if (err)
        return err;
    for (int64_t k = 0; k < matrix->size; k++)
        matrix->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_init_integer_double()’ allocates and initialises a
 * matrix in array format with integer, double precision coefficients.
 */
int mtxmatrix_array_init_integer_double(
    struct mtxmatrix_array * matrix,
    enum mtxsymmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data)
{
    int err = mtxmatrix_array_alloc(
        matrix, mtx_field_integer, mtx_double, symmetry, num_rows, num_columns);
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
 * ‘mtxmatrix_array_alloc_row_vector()’ allocates a row vector for a
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
 * ‘mtxmatrix_array_alloc_column_vector()’ allocates a column vector
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
 * ‘mtxmatrix_array_from_mtxfile()’ converts a matrix in Matrix Market
 * format to a matrix.
 */
int mtxmatrix_array_from_mtxfile(
    struct mtxmatrix_array * matrix,
    const struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from coordinate to array. */
    if (mtxfile->header.format != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxsymmetry symmetry;
    err = mtxfilesymmetry_to_mtxsymmetry(
        &symmetry, mtxfile->header.symmetry);
    if (err) return err;

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_real_single(
                matrix, symmetry, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_real_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_real_double(
                matrix, symmetry, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_complex_single(
                matrix, symmetry, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_complex_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_complex_double(
                matrix, symmetry, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            return mtxmatrix_array_init_integer_single(
                matrix, symmetry, mtxfile->size.num_rows, mtxfile->size.num_columns,
                mtxfile->data.array_integer_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxmatrix_array_init_integer_double(
                matrix, symmetry, mtxfile->size.num_rows, mtxfile->size.num_columns,
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
 * ‘mtxmatrix_array_to_mtxfile()’ converts a matrix to a matrix in
 * Matrix Market format.
 */
int mtxmatrix_array_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmatrix_array * matrix,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    enum mtxfilesymmetry symmetry;
    err = mtxfilesymmetry_from_mtxsymmetry(&symmetry, matrix->symmetry);
    if (err) return err;

    if (matrix->field == mtx_field_real) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_real_single(
                mtxfile, symmetry, matrix->num_rows, matrix->num_columns,
                matrix->data.real_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_real_double(
                mtxfile, symmetry, matrix->num_rows, matrix->num_columns,
                matrix->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_complex) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_complex_single(
                mtxfile, symmetry, matrix->num_rows, matrix->num_columns,
                matrix->data.complex_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_complex_double(
                mtxfile, symmetry, matrix->num_rows, matrix->num_columns,
                matrix->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (matrix->field == mtx_field_integer) {
        if (matrix->precision == mtx_single) {
            return mtxfile_init_matrix_array_integer_single(
                mtxfile, symmetry, matrix->num_rows, matrix->num_columns,
                matrix->data.integer_single);
        } else if (matrix->precision == mtx_double) {
            return mtxfile_init_matrix_array_integer_double(
                mtxfile, symmetry, matrix->num_rows, matrix->num_columns,
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
 * Nonzero rows and columns
 */

/**
 * ‘mtxmatrix_array_nzrows()’ counts the number of nonzero (non-empty)
 * matrix rows, and, optionally, fills an array with the row indices
 * of the nonzero (non-empty) matrix rows.
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
int mtxmatrix_array_nzrows(
    const struct mtxmatrix_array * matrix,
    int * num_nonzero_rows,
    int size,
    int * nonzero_rows)
{
    if (num_nonzero_rows)
        *num_nonzero_rows = matrix->num_rows;
    if (nonzero_rows) {
        if (size < matrix->num_rows)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        for (int i = 0; i < matrix->num_rows; i++)
            nonzero_rows[i] = i;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_nzcols()’ counts the number of nonzero (non-empty)
 * matrix columns, and, optionally, fills an array with the column
 * indices of the nonzero (non-empty) matrix columns.
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
int mtxmatrix_array_nzcols(
    const struct mtxmatrix_array * matrix,
    int * num_nonzero_columns,
    int size,
    int * nonzero_columns)
{
    if (num_nonzero_columns)
        *num_nonzero_columns = matrix->num_columns;
    if (nonzero_columns) {
        if (size < matrix->num_columns)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        for (int j = 0; j < matrix->num_columns; j++)
            nonzero_columns[j] = j;
    }
    return MTX_SUCCESS;
}


/*
 * Partitioning
 */

/**
 * ‘mtxmatrix_array_partition()’ partitions a matrix into blocks
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
int mtxmatrix_array_partition(
    struct mtxmatrix * dsts,
    const struct mtxmatrix_array * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    int err;
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    int num_parts = num_row_parts * num_col_parts;

    struct mtxfile mtxfile;
    err = mtxmatrix_array_to_mtxfile(&mtxfile, src, mtxfile_array);
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
        dsts[p].type = mtxmatrix_array;
        err = mtxmatrix_array_from_mtxfile(
            &dsts[p].storage.array, &dstmtxfiles[p]);
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
 * ‘mtxmatrix_array_join()’ joins together matrices representing
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
int mtxmatrix_array_join(
    struct mtxmatrix_array * dst,
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
        err = mtxmatrix_to_mtxfile(&srcmtxfiles[p], &srcs[p], mtxfile_array);
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

    err = mtxmatrix_array_from_mtxfile(dst, &dstmtxfile);
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

static int mtxmatrix_array_vectorise(
    struct mtxvector_array * vecx,
    const struct mtxmatrix_array * x)
{
    if (x->size > INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    vecx->field = x->field;
    vecx->precision = x->precision;
    vecx->size = x->size;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            vecx->data.real_single = x->data.real_single;
        } else if (x->precision == mtx_double) {
            vecx->data.real_double = x->data.real_double;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            vecx->data.complex_single = x->data.complex_single;
        } else if (x->precision == mtx_double) {
            vecx->data.complex_double = x->data.complex_double;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            vecx->data.integer_single = x->data.integer_single;
        } else if (x->precision == mtx_double) {
            vecx->data.integer_double = x->data.integer_double;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_swap()’ swaps values of two matrices,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_swap(
    struct mtxmatrix_array * x,
    struct mtxmatrix_array * y)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, x);
    if (err) return err;
    return mtxvector_array_swap(&vecx, &vecy);
}

/**
 * ‘mtxmatrix_array_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_copy(
    struct mtxmatrix_array * y,
    const struct mtxmatrix_array * x)
{
    struct mtxvector_array vecy;
    int err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    struct mtxvector_array vecx;
    err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_copy(&vecy, &vecx);
}

/**
 * ‘mtxmatrix_array_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_array_sscal(
    float a,
    struct mtxmatrix_array * x,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_sscal(a, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_array_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmatrix_array_dscal(
    double a,
    struct mtxmatrix_array * x,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_dscal(a, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_array_saxpy()’ adds a matrix to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_saxpy(
    float a,
    const struct mtxmatrix_array * x,
    struct mtxmatrix_array * y,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_saxpy(a, &vecx, &vecy, num_flops);
}

/**
 * ‘mtxmatrix_array_daxpy()’ adds a matrix to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_daxpy(
    double a,
    const struct mtxmatrix_array * x,
    struct mtxmatrix_array * y,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_daxpy(a, &vecx, &vecy, num_flops);
}

/**
 * ‘mtxmatrix_array_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_saypx(
    float a,
    struct mtxmatrix_array * y,
    const struct mtxmatrix_array * x,
    int64_t * num_flops)
{
    struct mtxvector_array vecy;
    int err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    struct mtxvector_array vecx;
    err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_saypx(a, &vecy, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_array_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_daypx(
    double a,
    struct mtxmatrix_array * y,
    const struct mtxmatrix_array * x,
    int64_t * num_flops)
{
    struct mtxvector_array vecy;
    int err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    struct mtxvector_array vecx;
    err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_daypx(a, &vecy, &vecx, num_flops);
}

/**
 * ‘mtxmatrix_array_sdot()’ computes the Frobenius inner product of
 * two matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_sdot(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    float * dot,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_sdot(&vecx, &vecy, dot, num_flops);
}

/**
 * ‘mtxmatrix_array_ddot()’ computes the Frobenius inner product of
 * two matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_ddot(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    double * dot,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_ddot(&vecx, &vecy, dot, num_flops);
}

/**
 * ‘mtxmatrix_array_cdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_cdotu(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_cdotu(&vecx, &vecy, dot, num_flops);
}

/**
 * ‘mtxmatrix_array_zdotu()’ computes the product of the transpose of
 * a complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_zdotu(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_zdotu(&vecx, &vecy, dot, num_flops);
}

/**
 * ‘mtxmatrix_array_cdotc()’ computes the Frobenius inner product of
 * two complex matrices in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_cdotc(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_cdotc(&vecx, &vecy, dot, num_flops);
}

/**
 * ‘mtxmatrix_array_zdotc()’ computes the Frobenius inner product of
 * two complex matrices in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxmatrix_array_zdotc(
    const struct mtxmatrix_array * x,
    const struct mtxmatrix_array * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    struct mtxvector_array vecy;
    err = mtxmatrix_array_vectorise(&vecy, y);
    if (err) return err;
    return mtxvector_array_zdotc(&vecx, &vecy, dot, num_flops);
}

/**
 * ‘mtxmatrix_array_snrm2()’ computes the Frobenius norm of a matrix in
 * single precision floating point.
 */
int mtxmatrix_array_snrm2(
    const struct mtxmatrix_array * x,
    float * nrm2,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_snrm2(&vecx, nrm2, num_flops);
}

/**
 * ‘mtxmatrix_array_dnrm2()’ computes the Frobenius norm of a matrix in
 * double precision floating point.
 */
int mtxmatrix_array_dnrm2(
    const struct mtxmatrix_array * x,
    double * nrm2,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_dnrm2(&vecx, nrm2, num_flops);
}

/**
 * ‘mtxmatrix_array_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_array_sasum(
    const struct mtxmatrix_array * x,
    float * asum,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_sasum(&vecx, asum, num_flops);
}

/**
 * ‘mtxmatrix_array_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxmatrix_array_dasum(
    const struct mtxmatrix_array * x,
    double * asum,
    int64_t * num_flops)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_dasum(&vecx, asum, num_flops);
}

/**
 * ‘mtxmatrix_array_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxmatrix_array_iamax(
    const struct mtxmatrix_array * x,
    int * iamax)
{
    struct mtxvector_array vecx;
    int err = mtxmatrix_array_vectorise(&vecx, x);
    if (err) return err;
    return mtxvector_array_iamax(&vecx, iamax);
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

#ifdef LIBMTX_HAVE_BLAS
enum CBLAS_TRANSPOSE mtxtransposition_to_cblas(
    enum mtxtransposition trans)
{
    if (trans == mtx_notrans) return CblasNoTrans;
    else if (trans == mtx_trans) return CblasTrans;
    else if (trans == mtx_conjtrans) return CblasConjTrans;
    else return -1;
}
#endif

/*
 * The operation counts below are taken from “Appendix C Operation
 * Counts for the BLAS and LAPACK” in Installation Guide for LAPACK by
 * Susan Blackford and Jack Dongarra, UT-CS-92-151, March, 1992.
 * Updated: June 30, 1999 (VERSION 3.0).
 */

static int64_t cblas_sgemv_num_flops(
    int64_t m, int64_t n, float alpha, float beta)
{
    return 2*m*n
        + (alpha == 1 || alpha == -1 ? 0 : m)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : m);
}

static int64_t cblas_sspmv_num_flops(
    int64_t n, float alpha, float beta)
{
    return 2*n*n
        + (alpha == 1 || alpha == -1 ? 0 : n)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : n);
}

static int64_t cblas_dgemv_num_flops(
    int64_t m, int64_t n, double alpha, double beta)
{
    return 2*m*n
        + (alpha == 1 || alpha == -1 ? 0 : m)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : m);
}

static int64_t cblas_dspmv_num_flops(
    int64_t n, double alpha, double beta)
{
    return 2*n*n
        + (alpha == 1 || alpha == -1 ? 0 : n)
        + (beta == 1 || beta == -1 || beta == 0 ? 0 : n);
}

static int64_t cblas_cgemv_num_flops(
    int64_t m, int64_t n, const float alpha[2], const float beta[2])
{
    return 8*m*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*m)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*m);
}

static int64_t cblas_chpmv_num_flops(
    int64_t n, const float alpha[2], const float beta[2])
{
    return 8*n*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*n)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*n);
}

static int64_t cblas_zgemv_num_flops(
    int64_t m, int64_t n, const double alpha[2], const double beta[2])
{
    return 8*m*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*m)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*m);
}

static int64_t cblas_zhpmv_num_flops(
    int64_t n, const double alpha[2], const double beta[2])
{
    return 8*n*n
        + ((alpha[0] == 1 && alpha[1] == 0) ||
           (alpha[0] == -1 && alpha[1] == 0) ? 0 : 6*n)
        + ((beta[0] == 1 && beta[1] == 0) ||
           (beta[0] == -1 && beta[1] == 0) ||
           (beta[0] == 0 && beta[1] == 0) ? 0 : 6*n);
}

/**
 * ‘mtxmatrix_array_sgemv()’ multiplies a matrix ‘A’, its transpose
 * ‘A'’ or its conjugate transpose ‘Aᴴ’ by a real scalar ‘alpha’ (‘α’)
 * and a vector ‘x’, before adding the result to another vector ‘y’
 * multiplied by another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x
 * + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
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
int mtxmatrix_array_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmatrix_array * A,
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
    if (A->num_rows == 0 || A->num_columns == 0)
        return MTX_SUCCESS;

    if (A->symmetry == mtx_unsymmetric) {
        if (A->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (A->precision == mtx_single) {
                    const float * Adata = A->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, beta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double * Adata = A->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, beta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float * Adata = A->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double * Adata = A->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_dgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_rows, A->num_columns, calpha, cbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_rows, A->num_columns, zalpha, zbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (A->precision == mtx_single) {
                    const int32_t * Adata = A->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const int64_t * Adata = A->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const int32_t * Adata = A->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const int64_t * Adata = A->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (A->field == mtx_field_real) {
            if (A->precision == mtx_single) {
                const float * Adata = A->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
                cblas_sspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_sspmv_num_flops(
                    A->num_rows, alpha, beta);
#else
                err = mtxvector_sscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double * Adata = A->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
                cblas_dspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_dspmv_num_flops(
                    A->num_rows, alpha, beta);
#else
                err = mtxvector_sscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (A->precision == mtx_single) {
                const int32_t * Adata = A->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                err = mtxvector_sscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else if (A->precision == mtx_double) {
                const int64_t * Adata = A->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                err = mtxvector_sscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_chpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, calpha,
                        (const float *) Adata, (const float *) xdata, 1,
                        cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_chpmv_num_flops(
                        A->num_rows, calpha, cbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zhpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, zalpha,
                        (const double *) Adata, (const double *) xdata, 1,
                        zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                        A->num_rows, zalpha, zbeta);
#else
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    err = mtxvector_sscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmatrix_array_sgemv()’ multiplies a matrix ‘A’, its transpose
 * ‘A'’ or its conjugate transpose ‘Aᴴ’ by a real scalar ‘alpha’ (‘α’)
 * and a vector ‘x’, before adding the result to another vector ‘y’
 * multiplied by another real scalar ‘beta’ (‘β’). That is, ‘y = α*A*x
 * + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
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
int mtxmatrix_array_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmatrix_array * A,
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
    if (A->num_rows == 0 || A->num_columns == 0)
        return MTX_SUCCESS;

    if (A->symmetry == mtx_unsymmetric) {
        if (A->field == mtx_field_real) {
            if (trans == mtx_notrans) {
                if (A->precision == mtx_single) {
                    const float * Adata = A->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, beta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double * Adata = A->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_rows, A->num_columns, alpha, beta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float * Adata = A->data.real_single;
                    const float * xdata = x_->data.real_single;
                    float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_sgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double * Adata = A->data.real_double;
                    const double * xdata = x_->data.real_double;
                    double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
                    cblas_dgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        alpha, Adata, A->num_columns, xdata, 1, beta, ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_dgemv_num_flops(
                        A->num_columns, A->num_rows, alpha, beta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_rows, A->num_columns, calpha, cbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_rows, A->num_columns, zalpha, zbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_cgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        calpha, (const float *) Adata, A->num_columns,
                        (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_cgemv_num_flops(
                        A->num_columns, A->num_rows, calpha, cbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zgemv(
                        CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                        zalpha, (const double *) Adata, A->num_columns,
                        (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zgemv_num_flops(
                        A->num_columns, A->num_rows, zalpha, zbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++) {
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (trans == mtx_notrans) {
                if (A->precision == mtx_single) {
                    const int32_t * Adata = A->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const int64_t * Adata = A->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[i] += alpha*Adata[k]*xdata[j];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const int32_t * Adata = A->data.integer_single;
                    const int32_t * xdata = x_->data.integer_single;
                    int32_t * ydata = y_->data.integer_single;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const int64_t * Adata = A->data.integer_double;
                    const int64_t * xdata = x_->data.integer_double;
                    int64_t * ydata = y_->data.integer_double;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        for (int j = 0; j < A->num_columns; j++, k++)
                            ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                    if (num_flops) *num_flops += 3*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (A->field == mtx_field_real) {
            if (A->precision == mtx_single) {
                const float * Adata = A->data.real_single;
                const float * xdata = x_->data.real_single;
                float * ydata = y_->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
                cblas_sspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_sspmv_num_flops(
                    A->num_rows, alpha, beta);
#else
                err = mtxvector_dscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double * Adata = A->data.real_double;
                const double * xdata = x_->data.real_double;
                double * ydata = y_->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
                cblas_dspmv(
                    CblasRowMajor, CblasUpper, A->num_rows,
                    alpha, Adata, xdata, 1, beta, ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_dspmv_num_flops(
                    A->num_rows, alpha, beta);
#else
                err = mtxvector_dscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else if (A->field == mtx_field_integer) {
            if (A->precision == mtx_single) {
                const int32_t * Adata = A->data.integer_single;
                const int32_t * xdata = x_->data.integer_single;
                int32_t * ydata = y_->data.integer_single;
                err = mtxvector_dscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else if (A->precision == mtx_double) {
                const int64_t * Adata = A->data.integer_double;
                const int64_t * xdata = x_->data.integer_double;
                int64_t * ydata = y_->data.integer_double;
                err = mtxvector_dscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i] += alpha*Adata[k++]*xdata[i];
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i] += alpha*Adata[k]*xdata[j];
                        ydata[j] += alpha*Adata[k]*xdata[i];
                    }
                }
                if (num_flops) *num_flops += 3*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (A->field == mtx_field_complex) {
            if (trans == mtx_notrans || trans == mtx_conjtrans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                    const float calpha[2] = {alpha, 0};
                    const float cbeta[2] = {beta, 0};
                    cblas_chpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, calpha,
                        (const float *) Adata, (const float *) xdata, 1,
                        cbeta, (float *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_chpmv_num_flops(
                        A->num_rows, calpha, cbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                    const double zalpha[2] = {alpha, 0};
                    const double zbeta[2] = {beta, 0};
                    cblas_zhpmv(
                        CblasRowMajor, CblasUpper, A->num_rows, zalpha,
                        (const double *) Adata, (const double *) xdata, 1,
                        zbeta, (double *) ydata, 1);
                    if (mtxblaserror()) return MTX_ERR_BLAS;
                    if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                        A->num_rows, zalpha, zbeta);
#else
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
#endif
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else if (trans == mtx_trans) {
                if (A->precision == mtx_single) {
                    const float (* Adata)[2] = A->data.complex_single;
                    const float (* xdata)[2] = x_->data.complex_single;
                    float (* ydata)[2] = y_->data.complex_single;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else if (A->precision == mtx_double) {
                    const double (* Adata)[2] = A->data.complex_double;
                    const double (* xdata)[2] = x_->data.complex_double;
                    double (* ydata)[2] = y_->data.complex_double;
                    err = mtxvector_dscal(beta, y, num_flops);
                    if (err) return err;
                    for (int i = 0, k = 0; i < A->num_rows; i++) {
                        ydata[i][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                        ydata[i][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        k++;
                        for (int j = i+1; j < A->num_columns; j++, k++) {
                            ydata[i][0] += alpha*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                            ydata[i][1] += alpha*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                            ydata[j][0] += alpha*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                            ydata[j][1] += alpha*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        }
                    }
                    if (num_flops) *num_flops += 10*A->num_entries;
                } else { return MTX_ERR_INVALID_PRECISION; }
            } else { return MTX_ERR_INVALID_TRANSPOSITION; }
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
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
 * the size of ‘x’ must equal the number of rows of ‘A’ and the size
 * of ‘y’ must equal the number of columns of ‘A’.
 */
int mtxmatrix_array_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmatrix_array * A,
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
    if (A->num_rows == 0 || A->num_columns == 0)
        return MTX_SUCCESS;

    if (A->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                cblas_cgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    alpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    zalpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, zalpha, zbeta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                cblas_cgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    alpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    zalpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, zalpha, zbeta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                cblas_cgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    alpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    zalpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, zalpha, zbeta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                cblas_chpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, alpha,
                    (const float *) Adata, (const float *) xdata, 1,
                    beta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_chpmv_num_flops(
                    A->num_rows, alpha, beta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
#endif
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                const double zalpha[2] = {alpha[0], alpha[1]};
                const double zbeta[2] = {beta[0], beta[1]};
                cblas_zhpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, zalpha,
                    (const double *) Adata, (const double *) xdata, 1,
                    zbeta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                    A->num_rows, zalpha, zbeta);
#else
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
#endif
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                err = mtxvector_cscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
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
    if (A->num_rows == 0 || A->num_columns == 0)
        return MTX_SUCCESS;

    if (A->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;

    if (A->symmetry == mtx_unsymmetric) {
        if (trans == mtx_notrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_cgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    calpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, calpha, cbeta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                cblas_zgemv(
                    CblasRowMajor, CblasNoTrans, A->num_rows, A->num_columns,
                    alpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1])-alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0])+alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_cgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    calpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, calpha, cbeta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                cblas_zgemv(
                    CblasRowMajor, CblasTrans, A->num_rows, A->num_columns,
                    alpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_cgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    calpha, (const float *) Adata, A->num_columns,
                    (const float *) xdata, 1, cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_cgemv_num_flops(
                    A->num_rows, A->num_columns, calpha, cbeta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                cblas_zgemv(
                    CblasRowMajor, CblasConjTrans, A->num_rows, A->num_columns,
                    alpha, (const double *) Adata, A->num_columns,
                    (const double *) xdata, 1, beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zgemv_num_flops(
                    A->num_rows, A->num_columns, alpha, beta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    for (int j = 0; j < A->num_columns; j++, k++) {
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1])-alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0])+alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
#endif
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_symmetric) {
        if (trans == mtx_notrans || trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_skew_symmetric) {
        /* TODO: allow skew-symmetric matrices */
        return MTX_ERR_INVALID_SYMMETRY;
    } else if (A->num_rows == A->num_columns && A->symmetry == mtx_hermitian) {
        if (trans == mtx_notrans || trans == mtx_conjtrans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
                const float calpha[2] = {alpha[0], alpha[1]};
                const float cbeta[2] = {beta[0], beta[1]};
                cblas_chpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, calpha,
                    (const float *) Adata, (const float *) xdata, 1,
                    cbeta, (float *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_chpmv_num_flops(
                    A->num_rows, calpha, cbeta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
#endif
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
                cblas_zhpmv(
                    CblasRowMajor, CblasUpper, A->num_rows, alpha,
                    (const double *) Adata, (const double *) xdata, 1,
                    beta, (double *) ydata, 1);
                if (mtxblaserror()) return MTX_ERR_BLAS;
                if (num_flops) *num_flops += cblas_zhpmv_num_flops(
                    A->num_rows, alpha, beta);
#else
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]+Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]-Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]-Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]+Adata[k][1]*xdata[i][1]);
                    }
                }
#endif
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (trans == mtx_trans) {
            if (A->precision == mtx_single) {
                const float (* Adata)[2] = A->data.complex_single;
                const float (* xdata)[2] = x_->data.complex_single;
                float (* ydata)[2] = y_->data.complex_single;
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else if (A->precision == mtx_double) {
                const double (* Adata)[2] = A->data.complex_double;
                const double (* xdata)[2] = x_->data.complex_double;
                double (* ydata)[2] = y_->data.complex_double;
                err = mtxvector_zscal(beta, y, num_flops);
                if (err) return err;
                for (int i = 0, k = 0; i < A->num_rows; i++) {
                    ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                    ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    k++;
                    for (int j = i+1; j < A->num_columns; j++, k++) {
                        ydata[i][0] += alpha[0]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]) - alpha[1]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]);
                        ydata[i][1] += alpha[0]*(Adata[k][0]*xdata[j][1]-Adata[k][1]*xdata[j][0]) + alpha[1]*(Adata[k][0]*xdata[j][0]+Adata[k][1]*xdata[j][1]);
                        ydata[j][0] += alpha[0]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]) - alpha[1]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]);
                        ydata[j][1] += alpha[0]*(Adata[k][0]*xdata[i][1]+Adata[k][1]*xdata[i][0]) + alpha[1]*(Adata[k][0]*xdata[i][0]-Adata[k][1]*xdata[i][1]);
                    }
                }
                if (num_flops) *num_flops += 14*A->num_entries;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else { return MTX_ERR_INVALID_SYMMETRY; }
    return MTX_SUCCESS;
}
