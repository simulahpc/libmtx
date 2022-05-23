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
 * Last modified: 2022-05-01
 *
 * Matrix Market files.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/cuthill_mckee.h>

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

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
 * ‘mtxfile_alloc()’ allocates storage for a Matrix Market file with
 * the given header line, comment lines and size line.
 *
 * ‘comments’ may be ‘NULL’, in which case it is ignored.
 */
int mtxfile_alloc(
    struct mtxfile * mtxfile,
    const struct mtxfileheader * header,
    const struct mtxfilecomments * comments,
    const struct mtxfilesize * size,
    enum mtxprecision precision)
{
    int err;
    err = mtxfileheader_copy(&mtxfile->header, header);
    if (err) return err;
    if (comments) {
        err = mtxfilecomments_copy(&mtxfile->comments, comments);
        if (err) return err;
    } else {
        err = mtxfilecomments_init(&mtxfile->comments);
        if (err) return err;
    }
    err = mtxfilesize_copy(&mtxfile->size, size);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    mtxfile->precision = precision;
    err = mtxfilesize_num_data_lines(size, header->symmetry, &mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    err = mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_free()’ frees storage allocated for a Matrix Market file.
 */
void mtxfile_free(
    struct mtxfile * mtxfile)
{
    mtxfiledata_free(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision);
    mtxfilecomments_free(&mtxfile->comments);
}

/**
 * ‘mtxfile_alloc_copy()’ allocates storage for a copy of a Matrix
 * Market file without initialising the underlying values.
 */
int mtxfile_alloc_copy(
    struct mtxfile * dst,
    const struct mtxfile * src)
{
    int err = mtxfileheader_copy(&dst->header, &src->header);
    if (err) return err;
    err = mtxfilecomments_copy(&dst->comments, &src->comments);
    if (err) return err;
    err = mtxfilesize_copy(&dst->size, &src->size);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    dst->precision = src->precision;
    dst->datasize = src->datasize;
    err = mtxfiledata_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, dst->datasize);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_copy()’ creates a copy of a Matrix Market file.
 */
int mtxfile_init_copy(
    struct mtxfile * dst,
    const struct mtxfile * src)
{
    int err = mtxfile_alloc_copy(dst, src);
    if (err) return err;
    err = mtxfiledata_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        src->datasize, 0, 0);
    if (err) {
        mtxfile_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Matrix array formats
 */

/**
 * ‘mtxfile_alloc_matrix_array()’ allocates a matrix in array format.
 */
int mtxfile_alloc_matrix_array(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns)
{
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (symmetry != mtxfile_general &&
        symmetry != mtxfile_symmetric &&
        symmetry != mtxfile_skew_symmetric &&
        symmetry != mtxfile_hermitian)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    mtxfile->header.object = mtxfile_matrix;
    mtxfile->header.format = mtxfile_array;
    mtxfile->header.field = field;
    mtxfile->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = num_columns;
    mtxfile->size.num_nonzeros = -1;
    mtxfile->precision = precision;
    int err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &mtxfile->datasize);
    if (err) return err;
    err = mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize);
    if (err) return err;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_array_real_single()’ allocates and initialises
 * a matrix in array format with real, single precision coefficients.
 */
int mtxfile_init_matrix_array_real_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const float * data)
{
    int err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_real, symmetry, mtx_single, num_rows, num_columns);
    if (err) return err;
    memcpy(mtxfile->data.array_real_single, data, mtxfile->datasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_array_real_double()’ allocates and initialises
 * a matrix in array format with real, double precision coefficients.
 */
int mtxfile_init_matrix_array_real_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const double * data)
{
    int err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_real, symmetry, mtx_double, num_rows, num_columns);
    if (err) return err;
    memcpy(mtxfile->data.array_real_double, data, mtxfile->datasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_array_complex_single()’ allocates and
 * initialises a matrix in array format with complex, single precision
 * coefficients.
 */
int mtxfile_init_matrix_array_complex_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const float (* data)[2])
{
    int err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_complex, symmetry, mtx_single, num_rows, num_columns);
    if (err) return err;
    memcpy(mtxfile->data.array_complex_single, data, mtxfile->datasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_array_complex_double()’ allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int mtxfile_init_matrix_array_complex_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const double (* data)[2])
{
    int err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_complex, symmetry, mtx_double, num_rows, num_columns);
    if (err) return err;
    memcpy(mtxfile->data.array_complex_double, data, mtxfile->datasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_array_integer_single()’ allocates and
 * initialises a matrix in array format with integer, single precision
 * coefficients.
 */
int mtxfile_init_matrix_array_integer_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const int32_t * data)
{
    int err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_integer, symmetry, mtx_single, num_rows, num_columns);
    if (err) return err;
    memcpy(mtxfile->data.array_integer_single, data, mtxfile->datasize * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_array_integer_double()’ allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int mtxfile_init_matrix_array_integer_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    const int64_t * data)
{
    int err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_integer, symmetry, mtx_double, num_rows, num_columns);
    if (err) return err;
    memcpy(mtxfile->data.array_integer_double, data, mtxfile->datasize * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Vector array formats
 */

/**
 * ‘mtxfile_alloc_vector_array()’ allocates a vector in array format.
 */
int mtxfile_alloc_vector_array(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows)
{
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    mtxfile->header.object = mtxfile_vector;
    mtxfile->header.format = mtxfile_array;
    mtxfile->header.field = field;
    mtxfile->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = -1;
    mtxfile->size.num_nonzeros = -1;
    mtxfile->precision = precision;
    mtxfile->datasize = num_rows;
    int err = mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_array_real_single()’ allocates and initialises
 * a vector in array format with real, single precision coefficients.
 */
int mtxfile_init_vector_array_real_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const float * data)
{
    int err = mtxfile_alloc_vector_array(mtxfile, mtxfile_real, mtx_single, num_rows);
    if (err) return err;
    memcpy(mtxfile->data.array_real_single, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_array_real_double()’ allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxfile_init_vector_array_real_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const double * data)
{
    int err = mtxfile_alloc_vector_array(mtxfile, mtxfile_real, mtx_double, num_rows);
    if (err) return err;
    memcpy(mtxfile->data.array_real_double, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_array_complex_single()’ allocates and
 * initialises a vector in array format with complex, single precision
 * coefficients.
 */
int mtxfile_init_vector_array_complex_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const float (* data)[2])
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_complex, mtx_single, num_rows);
    if (err) return err;
    memcpy(mtxfile->data.array_complex_single, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_array_complex_double()’ allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxfile_init_vector_array_complex_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const double (* data)[2])
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_complex, mtx_double, num_rows);
    if (err) return err;
    memcpy(mtxfile->data.array_complex_double, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_array_integer_single()’ allocates and
 * initialises a vector in array format with integer, single precision
 * coefficients.
 */
int mtxfile_init_vector_array_integer_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const int32_t * data)
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_integer, mtx_single, num_rows);
    if (err) return err;
    memcpy(mtxfile->data.array_integer_single, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_array_integer_double()’ allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_array_integer_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    const int64_t * data)
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_integer, mtx_double, num_rows);
    if (err) return err;
    memcpy(mtxfile->data.array_integer_double, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Matrix coordinate formats
 */

/**
 * ‘mtxfile_alloc_matrix_coordinate()’ allocates a matrix in
 * coordinate format.
 */
int mtxfile_alloc_matrix_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros)
{
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer &&
        field != mtxfile_pattern)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (symmetry != mtxfile_general &&
        symmetry != mtxfile_symmetric &&
        symmetry != mtxfile_skew_symmetric &&
        symmetry != mtxfile_hermitian)
        return MTX_ERR_INVALID_MTX_SYMMETRY;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    mtxfile->header.object = mtxfile_matrix;
    mtxfile->header.format = mtxfile_coordinate;
    mtxfile->header.field = field;
    mtxfile->header.symmetry = symmetry;
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = num_columns;
    mtxfile->size.num_nonzeros = num_nonzeros;
    mtxfile->precision = precision;
    mtxfile->datasize = num_nonzeros;
    return mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize);
}

/**
 * ‘mtxfile_init_matrix_coordinate_real_single()’ allocates and initialises
 * a matrix in coordinate format with real, single precision coefficients.
 */
int mtxfile_init_matrix_coordinate_real_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_single * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_real_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_coordinate_real_double()’ allocates and initialises
 * a matrix in coordinate format with real, double precision coefficients.
 */
int mtxfile_init_matrix_coordinate_real_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_double * data)
{
    int err;
    err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_real_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_coordinate_complex_single()’ allocates and
 * initialises a matrix in coordinate format with complex, single precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_complex_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_single * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_complex, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_complex_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_coordinate_complex_double()’ allocates and
 * initialises a matrix in coordinate format with complex, double precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_complex_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_double * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_complex, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_complex_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_coordinate_integer_single()’ allocates and
 * initialises a matrix in coordinate format with integer, single precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_integer_single(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_single * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_integer, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_integer_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_coordinate_integer_double()’ allocates and
 * initialises a matrix in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_integer_double(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_double * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_integer, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_integer_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_matrix_coordinate_pattern()’ allocates and
 * initialises a matrix in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_pattern(
    struct mtxfile * mtxfile,
    enum mtxfilesymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_pattern * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_pattern, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.matrix_coordinate_pattern,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxfile_alloc_vector_coordinate()’ allocates a vector in
 * coordinate format.
 */
int mtxfile_alloc_vector_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfilefield field,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_nonzeros)
{
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer &&
        field != mtxfile_pattern)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;
    mtxfile->header.object = mtxfile_vector;
    mtxfile->header.format = mtxfile_coordinate;
    mtxfile->header.field = field;
    mtxfile->header.symmetry = mtxfile_general;
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = -1;
    mtxfile->size.num_nonzeros = num_nonzeros;
    mtxfile->precision = precision;
    mtxfile->datasize = num_nonzeros;
    return mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_nonzeros);
}

/**
 * ‘mtxfile_init_vector_coordinate_real_single()’ allocates and initialises
 * a vector in coordinate format with real, single precision coefficients.
 */
int mtxfile_init_vector_coordinate_real_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_real, mtx_single, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_real_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_coordinate_real_double()’ allocates and initialises
 * a vector in coordinate format with real, double precision coefficients.
 */
int mtxfile_init_vector_coordinate_real_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_real, mtx_double, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_real_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_coordinate_complex_single()’ allocates and
 * initialises a vector in coordinate format with complex, single precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_complex_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_complex, mtx_single, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_complex_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_coordinate_complex_double()’ allocates and
 * initialises a vector in coordinate format with complex, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_complex_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_complex, mtx_double, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_complex_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_coordinate_integer_single()’ allocates and
 * initialises a vector in coordinate format with integer, single precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_integer_single(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_integer, mtx_single, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_integer_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_coordinate_integer_double()’ allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_integer_double(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_integer, mtx_double, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_integer_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_init_vector_coordinate_pattern()’ allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_pattern(
    struct mtxfile * mtxfile,
    int64_t num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_pattern, mtx_single, num_rows, num_nonzeros);
    if (err) return err;
    memcpy(mtxfile->data.vector_coordinate_pattern,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxfile_set_constant_real_single()’ sets every (nonzero) value of
 * a matrix or vector equal to a constant, single precision floating
 * point number.
 */
int mtxfile_set_constant_real_single(
    struct mtxfile * mtxfile,
    float a)
{
    return mtxfiledata_set_constant_real_single(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        mtxfile->datasize, 0, a);
}

/**
 * ‘mtxfile_set_constant_real_double()’ sets every (nonzero) value of
 * a matrix or vector equal to a constant, double precision floating
 * point number.
 */
int mtxfile_set_constant_real_double(
    struct mtxfile * mtxfile,
    double a)
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_set_constant_complex_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxfile_set_constant_complex_single(
    struct mtxfile * mtxfile,
    float a[2])
{
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_set_constant_integer_single()’ sets every (nonzero) value
 * of a matrix or vector equal to a constant integer.
 */
int mtxfile_set_constant_integer_single(
    struct mtxfile * mtxfile,
    int32_t a)
{
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxfile_read()’ reads a Matrix Market file from the given path.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_read(
    struct mtxfile * mtxfile,
    enum mtxprecision precision,
    const char * path,
    bool gzip,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int err;
    *lines_read = -1;
    *bytes_read = 0;

    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDIN_FILENO);
            if (fd == -1)
                return MTX_ERR_ERRNO;
            if ((f = fdopen(fd, "r")) == NULL) {
                close(fd);
                return MTX_ERR_ERRNO;
            }
        } else if ((f = fopen(path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        *lines_read = 0;
        err = mtxfile_fread(
            mtxfile, precision, f, lines_read, bytes_read, 0, NULL);
        if (err) {
            fclose(f);
            return err;
        }
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDIN_FILENO);
            if (fd == -1)
                return MTX_ERR_ERRNO;
            if ((f = gzdopen(fd, "r")) == NULL) {
                close(fd);
                return MTX_ERR_ERRNO;
            }
        } else if ((f = gzopen(path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        *lines_read = 0;
        err = mtxfile_gzread(
            mtxfile, precision, f, lines_read, bytes_read, 0, NULL);
        if (err) {
            gzclose(f);
            return err;
        }
        gzclose(f);
#else
        return MTX_ERR_ZLIB_NOT_SUPPORTED;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_fread()’ reads a Matrix Market file from a stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array that can
 * hold at least ‘line_max’ values of type ‘char’. This buffer is used
 * for reading lines from the stream. Otherwise, if ‘linebuf’ is
 * ‘NULL’, then a temporary buffer is allocated and used, and the
 * maximum line length is determined by calling ‘sysconf()’ with
 * ‘_SC_LINE_MAX’.
 */
int mtxfile_fread(
    struct mtxfile * mtxfile,
    enum mtxprecision precision,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    err = mtxfileheader_fread(
        &mtxfile->header, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_fread_comments(
        &mtxfile->comments, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfilesize_fread(
        &mtxfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object, mtxfile->header.format);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfiledata_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision, mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfiledata_fread(
        &mtxfile->data, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision,
        mtxfile->size.num_rows,
        mtxfile->size.num_columns,
        mtxfile->datasize, 0);
    if (err) {
        mtxfiledata_free(
            &mtxfile->data, mtxfile->header.object,
            mtxfile->header.format, mtxfile->header.field,
            precision);
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    mtxfile->precision = precision;
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfile_gzread()’ reads a Matrix Market file from a
 * gzip-compressed stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array that can
 * hold at least ‘line_max’ values of type ‘char’. This buffer is used
 * for reading lines from the stream. Otherwise, if ‘linebuf’ is
 * ‘NULL’, then a temporary buffer is allocated and used, and the
 * maximum line length is determined by calling ‘sysconf()’ with
 * ‘_SC_LINE_MAX’.
 */
int mtxfile_gzread(
    struct mtxfile * mtxfile,
    enum mtxprecision precision,
    gzFile f,
    int64_t * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    bool free_linebuf = !linebuf;
    if (!linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
        if (!linebuf)
            return MTX_ERR_ERRNO;
    }

    err = mtxfileheader_gzread(
        &mtxfile->header, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_gzread_comments(
        &mtxfile->comments, f, lines_read, bytes_read, line_max, linebuf);
    if (err) {
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfilesize_gzread(
        &mtxfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object, mtxfile->header.format);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfiledata_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision, mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfiledata_gzread(
        &mtxfile->data, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision,
        mtxfile->size.num_rows,
        mtxfile->size.num_columns,
        mtxfile->datasize, 0);
    if (err) {
        mtxfiledata_free(
            &mtxfile->data, mtxfile->header.object,
            mtxfile->header.format, mtxfile->header.field,
            precision);
        mtxfilecomments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    mtxfile->precision = precision;
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif

/**
 * ‘mtxfile_write()’ writes a Matrix Market file to the given path.
 * The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfile_write(
    const struct mtxfile * mtxfile,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    *bytes_written = 0;

    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDOUT_FILENO);
            if (fd == -1)
                return MTX_ERR_ERRNO;
            if ((f = fdopen(fd, "w")) == NULL) {
                close(fd);
                return MTX_ERR_ERRNO;
            }
        } else if ((f = fopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        err = mtxfile_fwrite(mtxfile, f, fmt, bytes_written);
        if (err) {
            fclose(f);
            return err;
        }
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDOUT_FILENO);
            if (fd == -1)
                return MTX_ERR_ERRNO;
            if ((f = gzdopen(fd, "w")) == NULL) {
                close(fd);
                return MTX_ERR_ERRNO;
            }
        } else if ((f = gzopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        err = mtxfile_gzwrite(mtxfile, f, fmt, bytes_written);
        if (err) {
            gzclose(f);
            return err;
        }
        gzclose(f);
#else
        return MTX_ERR_ZLIB_NOT_SUPPORTED;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_fwrite()’ writes a Matrix Market file to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfile_fwrite(
    const struct mtxfile * mtxfile,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err = mtxfileheader_fwrite(&mtxfile->header, f, bytes_written);
    if (err) return err;
    err = mtxfilecomments_fputs(&mtxfile->comments, f, bytes_written);
    if (err) return err;
    err = mtxfilesize_fwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err) return err;
    err = mtxfiledata_fwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize,
        f, fmt, bytes_written);
    if (err)
        return err;
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxfile_gzwrite()’ writes a Matrix Market file to a
 * gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxfile_gzwrite(
    const struct mtxfile * mtxfile,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err = mtxfileheader_gzwrite(&mtxfile->header, f, bytes_written);
    if (err) return err;
    err = mtxfilecomments_gzputs(&mtxfile->comments, f, bytes_written);
    if (err) return err;
    err = mtxfilesize_gzwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err) return err;
    err = mtxfiledata_gzwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize,
        f, fmt, bytes_written);
    if (err) return err;
    return MTX_SUCCESS;
}
#endif

/*
 * Transpose and conjugate transpose.
 */

/**
 * ‘mtxfile_transpose()’ tranposes a Matrix Market file.
 */
int mtxfile_transpose(
    struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object == mtxfile_matrix) {
        err = mtxfiledata_transpose(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, mtxfile->datasize);
        if (err) return err;
        err = mtxfilesize_transpose(&mtxfile->size);
        if (err)
            return err;
    } else if (mtxfile->header.object == mtxfile_vector) {
        return MTX_SUCCESS;
    } else {
        return MTX_ERR_INVALID_MTX_OBJECT;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_conjugate_transpose()’ tranposes and complex conjugates a
 * Matrix Market file.
 */
int mtxfile_conjugate_transpose(
    struct mtxfile * mtxfile);

/*
 * Sorting
 */

/**
 * ‘mtxfilesorting_str()’ is a string representing the sorting of a
 * matrix or vector in Matix Market format.
 */
const char * mtxfilesorting_str(
    enum mtxfilesorting sorting)
{
    switch (sorting) {
    case mtxfile_unsorted: return "unsorted";
    case mtxfile_permutation: return "permute";
    case mtxfile_row_major: return "row-major";
    case mtxfile_column_major: return "column-major";
    case mtxfile_morton: return "morton";
    default: return mtxstrerror(MTX_ERR_INVALID_SORTING);
    }
}

/**
 * ‘mtxfilesorting_parse()’ parses a string containing the ‘sorting’
 * of a Matrix Market file format header.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfilesorting_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘sorting’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfilesorting_parse(
    enum mtxfilesorting * sorting,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("unsorted", t, strlen("unsorted")) == 0) {
        t += strlen("unsorted");
        *sorting = mtxfile_unsorted;
    } else if (strncmp("permute", t, strlen("permute")) == 0) {
        t += strlen("permute");
        *sorting = mtxfile_permutation;
    } else if (strncmp("row-major", t, strlen("row-major")) == 0) {
        t += strlen("row-major");
        *sorting = mtxfile_row_major;
    } else if (strncmp("column-major", t, strlen("column-major")) == 0) {
        t += strlen("column-major");
        *sorting = mtxfile_column_major;
    } else if (strncmp("morton", t, strlen("morton")) == 0) {
        t += strlen("morton");
        *sorting = mtxfile_morton;
    } else {
        return MTX_ERR_INVALID_SORTING;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_SORTING;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_sort()’ sorts a Matrix Market file in a given order.
 *
 * The sorting order is determined by ‘sorting’. If the sorting order
 * is ‘mtxfile_unsorted’, nothing is done. If the sorting order is
 * ‘mtxfile_permutation’, then ‘perm’ must point to an array of ‘size’
 * integers that specify the sorting permutation. Note that the
 * sorting permutation uses 1-based indexing.
 *
 * For a vector or matrix in coordinate format, the nonzero values are
 * sorted in the specified order. For Matrix Market files in array
 * format, this operation does nothing.
 *
 * ‘perm’ is ignored if it is ‘NULL’. Otherwise, it must point to an
 * array of length ‘size’, which is used to store the permutation of
 * the Matrix Market entries. ‘size’ must therefore be at least equal
 * to the number of data lines in the Matrix Market file ‘mtx’.
 */
int mtxfile_sort(
    struct mtxfile * mtxfile,
    enum mtxfilesorting sorting,
    int64_t size,
    int64_t * perm)
{
    if (perm && (size < 0 || size > mtxfile->datasize))
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (sorting == mtxfile_unsorted) {
        if (!perm)
            return MTX_SUCCESS;
        for (int64_t k = 0; k < mtxfile->datasize; k++)
            perm[k] = k+1;
        return MTX_SUCCESS;
    } else if (sorting == mtxfile_permutation) {
        return mtxfiledata_permute(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, mtxfile->datasize, perm);
    } else if (sorting == mtxfile_row_major) {
        return mtxfiledata_sort_row_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, mtxfile->datasize, perm);
    } else if (sorting == mtxfile_column_major) {
        return mtxfiledata_sort_column_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, mtxfile->datasize, perm);
    } else if (sorting == mtxfile_morton) {
        return mtxfiledata_sort_morton(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, mtxfile->datasize, perm);
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
}

/**
 * ‘mtxfile_compact()’ compacts a Matrix Market file in coordinate
 * format by merging adjacent, duplicate entries.
 *
 * For a matrix or vector in array format, this does nothing.
 *
 * The number of nonzero matrix or vector entries,
 * ‘mtxfile->size.num_nonzeros’, is updated to reflect entries that
 * were removed as a result of compacting. However, the underlying
 * storage is not changed or reallocated. This may result in large
 * amounts of unused memory, if a large number of entries were
 * removed. If necessary, it is possible to allocate new storage, copy
 * the compacted data, and, finally, free the old storage.
 *
 * If ‘perm’ is not ‘NULL’, then it must point to an array of length
 * ‘size’. The ‘i’th entry of ‘perm’ is used to store the index of the
 * corresponding data line in the compacted array that the ‘i’th data
 * line was moved to or merged with. Note that the indexing is
 * 1-based.
 */
int mtxfile_compact(
    struct mtxfile * mtxfile,
    int64_t size,
    int64_t * perm)
{
    int err;
    if (mtxfile->header.format == mtxfile_array)
        return MTX_SUCCESS;
    int64_t num_data_lines = mtxfile->size.num_nonzeros;
    if (perm && (size < 0 || size > num_data_lines))
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t outsize;
    err = mtxfiledata_compact(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
        mtxfile->size.num_columns, mtxfile->size.num_nonzeros, perm,
        &outsize);
    if (err)
        return err;
    mtxfile->size.num_nonzeros = outsize;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_assemble()’ assembles a Matrix Market file in coordinate
 * format by merging duplicate entries. The file may optionally be
 * sorted at the same time.
 *
 * For a matrix or vector in array format, this does nothing.
 *
 * The number of nonzero matrix or vector entries,
 * ‘mtxfile->size.num_nonzeros’, is updated to reflect entries that
 * were removed as a result of compacting. However, the underlying
 * storage is not changed or reallocated. This may result in large
 * amounts of unused memory, if a large number of entries were
 * removed. If necessary, it is possible to allocate new storage, copy
 * the compacted data, and, finally, free the old storage.
 *
 * If ‘perm’ is not ‘NULL’, then it must point to an array of length
 * ‘size’. The ‘i’th entry of ‘perm’ is used to store the index of the
 * corresponding data line in the sorted and compacted array that the
 * ‘i’th data line was moved to or merged with. Note that the indexing
 * is 1-based.
 */
int mtxfile_assemble(
    struct mtxfile * mtxfile,
    enum mtxfilesorting sorting,
    int64_t size,
    int64_t * perm)
{
    int err;
    if (sorting == mtxfile_unsorted) {
        /* TODO: It may be desirable to perform assembly to remove
         * duplicates without otherwise changing the original order of
         * the data lines. To do this, we should first sort, for
         * example, in row major order, to obtain a sorting
         * permutation. Then perform the compaction by accessing the
         * data indirectly through the sorting permutation. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    }
    err = mtxfile_sort(mtxfile, sorting, size, perm);
    if (err)
        return err;
    err = mtxfile_compact(mtxfile, size, perm);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */


/**
 * ‘mtxfile_partition_nonzeros()’ partitions the nonzeros of a Matrix
 * Market file.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’ and ‘blksize’.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If it is not ‘NULL’, then the array ‘partsptr’ must contain enough
 * storage for ‘num_parts+1’ values of type ‘int64_t’. If successful,
 * ‘partsptr’ will contain offsets to the first element belonging to
 * each part.
 */
int mtxfile_partition_nonzeros(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning parttype,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    int * dstpart,
    int64_t * partsptr)
{
    int64_t * idx = malloc(mtxfile->datasize * sizeof(int64_t));
    if (!idx) return MTX_ERR_ERRNO;
    for (int64_t k = 0; k < mtxfile->datasize; k++) idx[k] = k;
    int err = partition_int64(
        parttype, mtxfile->datasize, num_parts, partsizes, blksize,
        mtxfile->datasize, sizeof(int64_t), idx, dstpart);
    if (err) { free(idx); return err; }
    if (partsptr) {
        for (int p = 0; p <= num_parts; p++) partsptr[p] = 0;
        for (int64_t k = 0; k < mtxfile->datasize; k++) partsptr[dstpart[k]+1]++;
        for (int p = 1; p <= num_parts; p++) partsptr[p] += partsptr[p-1];
    }
    free(idx);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_partition_rowwise()’ partitions the entries of a Matrix
 * Market file according to a given row partitioning.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’ and ‘blksize’.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If it is not ‘NULL’, then the array ‘partsptr’ must contain enough
 * storage for ‘num_parts+1’ values of type ‘int64_t’. If successful,
 * ‘partsptr’ will contain offsets to the first element belonging to
 * each part.
 */
int mtxfile_partition_rowwise(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning parttype,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    int * dstpart,
    int64_t * partsptr)
{
    return mtxfiledata_partition_rowwise(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        mtxfile->size.num_rows, mtxfile->size.num_columns, 0, mtxfile->datasize,
        parttype, num_parts, partsizes, blksize, dstpart, partsptr);
}

/**
 * ‘mtxfile_partition_columnwise()’ partitions the entries of a Matrix
 * Market file according to a given column partitioning.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘parttype’, ‘num_parts’, ‘partsizes’ and ‘blksize’.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If it is not ‘NULL’, then the array ‘partsptr’ must contain enough
 * storage for ‘num_parts+1’ values of type ‘int64_t’. If successful,
 * ‘partsptr’ will contain offsets to the first element belonging to
 * each part.
 */
int mtxfile_partition_columnwise(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning parttype,
    int num_parts,
    const int64_t * partsizes,
    int64_t blksize,
    int * dstpart,
    int64_t * partsptr)
{
    return mtxfiledata_partition_columnwise(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        mtxfile->size.num_rows, mtxfile->size.num_columns, 0, mtxfile->datasize,
        parttype, num_parts, partsizes, blksize, dstpart, partsptr);
}

/**
 * ‘mtxfile_partition_2d()’ partitions a matrix in Matrix Market
 * format according to given row and column partitionings.
 *
 * The number of parts is equal to the product of ‘num_row_parts’ and
 * ‘num_column_parts’.
 *
 * See ‘partition_int64()’ for an explanation of the meaning of the
 * arguments ‘rowparttype’, ‘num_row_parts’, ‘rowpartsizes’ and
 * ‘rowblksize’, and so on.
 *
 * The array ‘dstpart’ must contain enough storage for
 * ‘mtxfile->datasize’ values of type ‘int’. If successful, ‘dstpart’
 * is used to store the part number assigned to the matrix or vector
 * nonzeros.
 *
 * If it is not ‘NULL’, then the array ‘partsptr’ must contain enough
 * storage for ‘num_row_parts*num_column_parts+1’ values of type
 * ‘int64_t’. If successful, ‘partsptr’ will contain offsets to the
 * first element belonging to each part.
 */
int mtxfile_partition_2d(
    const struct mtxfile * mtxfile,
    enum mtxpartitioning rowparttype,
    int num_row_parts,
    const int64_t * rowpartsizes,
    int64_t rowblksize,
    enum mtxpartitioning colparttype,
    int num_column_parts,
    const int64_t * colpartsizes,
    int64_t colblksize,
    int * dstpart,
    int64_t * partsptr)
{
    return mtxfiledata_partition_2d(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        mtxfile->size.num_rows, mtxfile->size.num_columns, 0, mtxfile->datasize,
        rowparttype, num_row_parts, rowpartsizes, rowblksize,
        colparttype, num_column_parts, colpartsizes, colblksize,
        dstpart, partsptr);
}

/**
 * ‘mtxfile_partition2()’ partitions the entries of a Matrix Market
 * file, and, optionally, also partitions the rows and columns of the
 * underlying matrix or vector.
 *
 * The type of partitioning to perform is determined by
 * ‘matrixparttype’.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_nonzeros’, the
 *    nonzeros of the underlying matrix or vector are partitioned as a
 *    one-dimensional array. The nonzeros are partitioned into
 *    ‘num_nz_parts’ parts according to the partitioning
 *    ‘nzparttype’. If ‘nzparttype’ is ‘mtx_block’, then ‘nzpartsizes’
 *    may be used to specify the size of each part. If ‘nzparttype’ is
 *    ‘mtx_block_cyclic’, then ‘nzblksize’ is used to specify the
 *    block size.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_rows’, the nonzeros
 *    of the underlying matrix or vector are partitioned rowwise.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_columns’, the
 *    nonzeros of the underlying matrix are partitioned columnwise.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_2d’, the nonzeros of
 *    the underlying matrix are partitioned in rectangular blocks
 *    according to the partitioning of the rows and columns.
 *
 *  - If ‘matrixparttype’ is ‘mtx_matrixparttype_metis’, then the rows
 *    and columns of the underlying matrix are partitioned by the
 *    METIS graph partitioner, and the matrix nonzeros are partitioned
 *    accordingly.
 */
int mtxfile_partition2(
    struct mtxfile * mtxfile,
    enum mtxmatrixparttype matrixparttype,
    enum mtxpartitioning nzparttype,
    int num_nz_parts,
    const int64_t * nzpartsizes,
    int64_t nzblksize,
    enum mtxpartitioning rowparttype,
    int num_row_parts,
    const int64_t * rowpartsizes,
    int64_t rowblksize,
    enum mtxpartitioning colparttype,
    int num_column_parts,
    const int64_t * colpartsizes,
    int64_t colblksize,
    int * dstpart,
    int64_t * partsptr)
{
    if (matrixparttype == mtx_matrixparttype_nonzeros) {
        return mtxfile_partition_nonzeros(
            mtxfile, nzparttype, num_nz_parts, nzpartsizes, nzblksize,
            dstpart, partsptr);
    } else if (matrixparttype == mtx_matrixparttype_rows) {
        return mtxfile_partition_rowwise(
            mtxfile, rowparttype, num_row_parts, rowpartsizes, rowblksize,
            dstpart, partsptr);
    } else if (matrixparttype == mtx_matrixparttype_columns) {
        return mtxfile_partition_columnwise(
            mtxfile, colparttype, num_column_parts, colpartsizes, colblksize,
            dstpart, partsptr);
    } else if (matrixparttype == mtx_matrixparttype_2d) {
        return mtxfile_partition_2d(
            mtxfile, rowparttype, num_row_parts, rowpartsizes, rowblksize,
            colparttype, num_column_parts, colpartsizes, colblksize,
            dstpart, partsptr);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_partition()’ partitions a Matrix Market file according to
 * the given row and column partitions.
 *
 * The partitions ‘rowpart’ or ‘colpart’ are allowed to be ‘NULL’, in
 * which case a trivial, singleton partition is used for the rows or
 * columns, respectively.
 *
 * Otherwise, ‘rowpart’ and ‘colpart’ must partition the rows and
 * columns of the matrix or vector ‘src’, respectively. That is,
 * ‘rowpart->size’ must be equal to ‘src->size.num_rows’, and
 * ‘colpart->size’ must be equal to ‘src->size.num_columns’.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P*Q’ values of type ‘struct mtxfile’, where ‘P’ is the number of
 * row parts, ‘rowpart->num_parts’, and ‘Q’ is the number of column
 * parts, ‘colpart->num_parts’. Note that the ‘r’th part corresponds
 * to a row part ‘p’ and column part ‘q’, such that ‘r=p*Q+q’. Thus,
 * the ‘r’th entry of ‘dsts’ is the submatrix corresponding to the
 * ‘p’th row and ‘q’th column of the 2D partitioning.
 *
 * The user is responsible for freeing storage allocated for each
 * Matrix Market file in the ‘dsts’ array.
 */
int mtxfile_partition(
    struct mtxfile * dsts,
    const struct mtxfile * src,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    int err;
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    if (rowpart &&
        ((src->size.num_rows == -1 && rowpart->size != 1) ||
         (src->size.num_rows >= 0  && rowpart->size != src->size.num_rows)))
        return MTX_ERR_INCOMPATIBLE_PARTITION;
    if (colpart &&
        ((src->size.num_columns == -1 && colpart->size != 1) ||
         (src->size.num_columns >= 0  && colpart->size != src->size.num_columns)))
        return MTX_ERR_INCOMPATIBLE_PARTITION;

    int * part_per_data_line = malloc(src->datasize * sizeof(int));
    if (!part_per_data_line)
        return MTX_ERR_ERRNO;
    int64_t * localrowidx = NULL;
    int64_t * localcolidx = NULL;
    if (src->header.format == mtxfile_array) {
        localrowidx = malloc(src->datasize * sizeof(int64_t));
        if (!localrowidx) {
            free(part_per_data_line);
            return MTX_ERR_ERRNO;
        }
        localcolidx = malloc(src->datasize * sizeof(int64_t));
        if (!localcolidx) {
            free(localrowidx);
            free(part_per_data_line);
            return MTX_ERR_ERRNO;
        }
    }

    /* 1. Assign each data line to its row and column part */
    err = mtxfiledata_partition(
        &src->data, src->header.object, src->header.format,
        src->header.field, src->precision,
        src->size.num_rows, src->size.num_columns,
        0, src->datasize, rowpart, colpart, part_per_data_line,
        localrowidx, localcolidx);
    if (err) {
        if (src->header.format == mtxfile_array) free(localcolidx);
        if (src->header.format == mtxfile_array) free(localrowidx);
        free(part_per_data_line);
        return err;
    }

    /* 2. Count the number of elements in each part */
    int num_parts = num_row_parts * num_col_parts;
    int64_t * num_data_lines_per_part = malloc(num_parts * sizeof(int64_t));
    if (!num_data_lines_per_part) {
        if (src->header.format == mtxfile_array) free(localcolidx);
        if (src->header.format == mtxfile_array) free(localrowidx);
        free(part_per_data_line);
        return MTX_ERR_ERRNO;
    }
    int64_t * data_lines_per_part_ptr = malloc((num_parts+1) * sizeof(int64_t));
    if (!data_lines_per_part_ptr) {
        free(num_data_lines_per_part);
        if (src->header.format == mtxfile_array) free(localcolidx);
        if (src->header.format == mtxfile_array) free(localrowidx);
        free(part_per_data_line);
        return MTX_ERR_ERRNO;
    }
    for (int p = 0; p < num_parts; p++)
        num_data_lines_per_part[p] = 0;
    for (int64_t k = 0; k < src->datasize; k++) {
        int part = part_per_data_line[k];
        num_data_lines_per_part[part]++;
    }
    data_lines_per_part_ptr[0] = 0;
    for (int p = 1; p <= num_parts; p++) {
        data_lines_per_part_ptr[p] =
            data_lines_per_part_ptr[p-1] +
            num_data_lines_per_part[p-1];
    }

    /* Create a copy of the data that can be sorted */
    union mtxfiledata data;
    err = mtxfiledata_alloc(
        &data, src->header.object, src->header.format,
        src->header.field, src->precision, src->datasize);
    if (err) {
        free(data_lines_per_part_ptr);
        free(num_data_lines_per_part);
        if (src->header.format == mtxfile_array) free(localcolidx);
        if (src->header.format == mtxfile_array) free(localrowidx);
        free(part_per_data_line);
        return err;
    }
    err = mtxfiledata_copy(
        &data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        src->datasize, 0, 0);
    if (err) {
        mtxfiledata_free(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision);
        free(data_lines_per_part_ptr);
        free(num_data_lines_per_part);
        if (src->header.format == mtxfile_array) free(localcolidx);
        if (src->header.format == mtxfile_array) free(localrowidx);
        free(part_per_data_line);
        return err;
    }

    /* Allocate sorting keys */
    uint64_t * sortkeys = malloc(src->datasize * sizeof(uint64_t));
    if (!sortkeys) {
        mtxfiledata_free(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision);
        free(data_lines_per_part_ptr);
        free(num_data_lines_per_part);
        if (src->header.format == mtxfile_array) free(localcolidx);
        if (src->header.format == mtxfile_array) free(localrowidx);
        free(part_per_data_line);
        return MTX_ERR_ERRNO;
    }

    if (src->header.format == mtxfile_array) {
        /* For a matrix in array format, we need to sort by part
         * number, and then in row major order using the local row and
         * column indices within each part. */
        for (int64_t k = 0; k < src->datasize; k++) {
            int r = part_per_data_line[k];
            int q = r % num_col_parts;
            int64_t num_columns =
                colpart ? colpart->part_sizes[q] : src->size.num_columns;
            sortkeys[k] = data_lines_per_part_ptr[r]
                + ((localrowidx ? localrowidx[k] : 0) *
                   (num_columns >= 0 ? num_columns : 1))
                + (localcolidx ? localcolidx[k] : 0);
        }
        free(localcolidx);
        free(localrowidx);

    } else if (src->header.format == mtxfile_coordinate) {
        /* For a matrix in coordinate format, simply sort the nonzeros
         * by their part numbers. Since the sorting is stable, the
         * order within each part remains the same as in the original
         * matrix or vector. */
        for (int64_t k = 0; k < src->datasize; k++) {
            int part = part_per_data_line[k];
            sortkeys[k] = part;
        }

        /* Find row and column permutations induced by partitioning */
        int64_t * rowperm64 = NULL;
        if (rowpart && src->size.num_rows >= 0) {
            rowperm64 = malloc(src->size.num_rows * sizeof(int64_t));
            if (!rowperm64) {
                free(sortkeys);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                free(data_lines_per_part_ptr);
                free(num_data_lines_per_part);
                free(part_per_data_line);
                return MTX_ERR_ERRNO;
            }
            for (int i = 0; i < src->size.num_rows; i++)
                rowperm64[i] = i;
            err = mtxpartition_assign(
                rowpart, src->size.num_rows, rowperm64, NULL, rowperm64);
            if (err) {
                free(rowperm64);
                free(sortkeys);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                free(data_lines_per_part_ptr);
                free(num_data_lines_per_part);
                free(part_per_data_line);
                return err;
            }
        }
        int64_t * colperm64 = NULL;
        if (colpart && src->size.num_columns >= 0) {
            colperm64 = malloc(src->size.num_columns * sizeof(int64_t));
            if (!colperm64) {
                if (rowpart && src->size.num_rows >= 0) free(rowperm64);
                free(sortkeys);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                free(data_lines_per_part_ptr);
                free(num_data_lines_per_part);
                free(part_per_data_line);
                return MTX_ERR_ERRNO;
            }
            for (int j = 0; j < src->size.num_columns; j++)
                colperm64[j] = j;
            err = mtxpartition_assign(
                colpart, src->size.num_columns, colperm64, NULL, colperm64);
            if (err) {
                free(colperm64);
                if (rowpart && src->size.num_rows >= 0) free(rowperm64);
                free(sortkeys);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                free(data_lines_per_part_ptr);
                free(num_data_lines_per_part);
                free(part_per_data_line);
                return err;
            }
        }

        int * rowperm = (int *) rowperm64;
        if (rowperm) {
            for (int i = 0; i < src->size.num_rows; i++)
                rowperm[i] = rowperm64[i]+1;
        }
        int * colperm = (int *) colperm64;
        if (colperm) {
            for (int i = 0; i < src->size.num_columns; i++)
                colperm[i] = colperm64[i]+1;
        }

        /* Convert global row and column numbers to local, partwise
         * row and column numbers. */
        err = mtxfiledata_reorder(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision, src->datasize, 0,
            src->size.num_rows, rowperm, src->size.num_columns, colperm);
        if (err) {
            if (colpart && src->size.num_columns >= 0) free(colperm64);
            if (rowpart && src->size.num_rows >= 0) free(rowperm64);
            free(sortkeys);
            mtxfiledata_free(
                &data, src->header.object, src->header.format,
                src->header.field, src->precision);
            free(data_lines_per_part_ptr);
            free(num_data_lines_per_part);
            free(part_per_data_line);
            return err;
        }

        if (colpart && src->size.num_columns >= 0) free(colperm64);
        if (rowpart && src->size.num_rows >= 0) free(rowperm64);
    }

    free(part_per_data_line);

    /* Sort the data by the keys */
    err = mtxfiledata_sort_keys(
        &data, src->header.object, src->header.format,
        src->header.field, src->precision,
        src->size.num_rows, src->size.num_columns,
        src->datasize, sortkeys, NULL);
    if (err) {
        free(sortkeys);
        mtxfiledata_free(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision);
        free(data_lines_per_part_ptr);
        free(num_data_lines_per_part);
        return err;
    }

    free(sortkeys);

    /* 3. Create a submatrix or -vector for each part */
    for (int p = 0; p < num_row_parts; p++) {
        for (int q = 0; q < num_col_parts; q++) {
            int r = p * num_col_parts + q;
            int64_t N = data_lines_per_part_ptr[r+1] - data_lines_per_part_ptr[r];

            struct mtxfilesize size;
            size.num_rows = rowpart
                ? rowpart->part_sizes[p] : src->size.num_rows;
            size.num_columns = colpart
                ? colpart->part_sizes[q] : src->size.num_columns;
            size.num_nonzeros = src->size.num_nonzeros >= 0 ? N : -1;

            err = mtxfile_alloc(
                &dsts[r], &src->header, &src->comments,
                &size, src->precision);
            if (err) {
                for (int s = r-1; s >= 0; s--)
                    mtxfile_free(&dsts[s]);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                free(data_lines_per_part_ptr);
                free(num_data_lines_per_part);
                return err;
            }

            err = mtxfiledata_copy(
                &dsts[r].data, &data,
                dsts[r].header.object, dsts[r].header.format,
                dsts[r].header.field, dsts[r].precision,
                N, 0, data_lines_per_part_ptr[r]);
            if (err) {
                for (int s = r; s >= 0; s--)
                    mtxfile_free(&dsts[s]);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                free(data_lines_per_part_ptr);
                free(num_data_lines_per_part);
                return err;
            }
        }
    }

    mtxfiledata_free(
        &data, src->header.object, src->header.format,
        src->header.field, src->precision);
    free(data_lines_per_part_ptr);
    free(num_data_lines_per_part);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_split()’ splits a Matrix Market file into several files
 * according to a given partition of the underlying (nonzero) matrix
 * or vector elements.
 *
 * The partitioning of the matrix or vector elements is specified by
 * the array ‘parts’. The length of the ‘parts’ array is given by
 * ‘size’, which must match the number of (nonzero) matrix or vector
 * elements in ‘src’. Each entry in the array is an integer in the
 * range ‘[0, num_parts)’ designating the part to which the
 * corresponding nonzero element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxfile’. If successful, then ‘dsts[p]’ points to a
 * matrix market file consisting of (nonzero) elements from ‘src’ that
 * belong to the ‘p’th part, according to the ‘parts’ array.
 *
 * If ‘src’ is a matrix (or vector) in coordinate format, then each of
 * the matrices or vectors in ‘dsts’ is also a matrix (or vector) in
 * coordinate format with the same number of rows and columns as
 * ‘src’. In this case, the arguments ‘num_rows_per_part’ and
 * ‘num_columns_per_part’ are not used and may be set to ‘NULL’.
 *
 * Otherwise, if ‘src’ is a matrix (or vector) in array format, then
 * the arrays ‘num_rows_per_part’ and ‘num_columns_per_part’ (both of
 * length ‘num_parts’) are used to specify the dimensions of each
 * matrix (or vector) in ‘dsts’. For a given part ‘p’, the number of
 * matrix (or vector) elements assigned to that part must be equal to
 * the product of ‘num_rows_per_part[p]’ and
 * ‘num_columns_per_part[p]’.
 *
 * The user is responsible for freeing storage allocated for each
 * Matrix Market file in the ‘dsts’ array.
 */
int mtxfile_split(
    int num_parts,
    struct mtxfile ** dsts,
    const struct mtxfile * src,
    int64_t size,
    int * parts,
    const int64_t * num_rows_per_part,
    const int64_t * num_columns_per_part)
{
    if (size != src->datasize) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    for (int64_t k = 0; k < size; k++) {
        if (parts[k] < 0 || parts[k] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* create a copy of the data that can be sorted */
    union mtxfiledata data;
    int err = mtxfiledata_alloc(
        &data, src->header.object, src->header.format,
        src->header.field, src->precision, src->datasize);
    if (err) return err;
    err = mtxfiledata_copy(
        &data, &src->data, src->header.object, src->header.format,
        src->header.field, src->precision, src->datasize, 0, 0);
    if (err) {
        mtxfiledata_free(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision);
        return err;
    }

    /* sort by part number */
    err = mtxfiledata_sort_int(
        &data, src->header.object, src->header.format,
        src->header.field, src->precision,
        src->size.num_rows, src->size.num_columns,
        src->datasize, parts, NULL);
    if (err) {
        mtxfiledata_free(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision);
        return err;
    }

    /* count the number of elements in each part */
    int64_t * partsizes = malloc(num_parts * sizeof(int64_t));
    if (!partsizes) {
        mtxfiledata_free(
            &data, src->header.object, src->header.format,
            src->header.field, src->precision);
        return MTX_ERR_ERRNO;
    }
    for (int p = 0; p < num_parts; p++)
        partsizes[p] = 0;
    for (int64_t k = 0; k < src->datasize; k++)
        partsizes[parts[k]]++;

    /* extract a submatrix or -vector for each part */
    int64_t srcoffset = 0;
    for (int p = 0; p < num_parts; p++) {
        struct mtxfilesize dstsize;
        if (src->header.object == mtxfile_matrix) {
            if (src->header.format == mtxfile_coordinate) {
                dstsize.num_rows = src->size.num_rows;
                dstsize.num_columns = src->size.num_columns;
                dstsize.num_nonzeros = partsizes[p];
            } else if (src->header.format == mtxfile_array) {
                dstsize.num_rows = num_rows_per_part[p];
                dstsize.num_columns = num_columns_per_part[p];
                dstsize.num_nonzeros = -1;
                if (partsizes[p] != num_rows_per_part[p]*num_columns_per_part[p]) {
                    for (int s = p-1; s >= 0; s--) mtxfile_free(dsts[s]);
                    free(partsizes);
                    mtxfiledata_free(
                        &data, src->header.object, src->header.format,
                        src->header.field, src->precision);
                    return MTX_ERR_INVALID_MTX_SIZE;
                }
            } else {
                for (int s = p-1; s >= 0; s--) mtxfile_free(dsts[s]);
                free(partsizes);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                return MTX_ERR_INVALID_MTX_FORMAT;
            }
        } else if (src->header.object == mtxfile_vector) {
            if (src->header.format == mtxfile_coordinate) {
                dstsize.num_rows = src->size.num_rows;
                dstsize.num_columns = -1;
                dstsize.num_nonzeros = partsizes[p];
            } else if (src->header.format == mtxfile_array) {
                dstsize.num_rows = num_rows_per_part[p];
                dstsize.num_columns = -1;
                dstsize.num_nonzeros = -1;
                if (partsizes[p] != num_rows_per_part[p]) {
                    for (int s = p-1; s >= 0; s--) mtxfile_free(dsts[s]);
                    free(partsizes);
                    mtxfiledata_free(
                        &data, src->header.object, src->header.format,
                        src->header.field, src->precision);
                    return MTX_ERR_INVALID_MTX_SIZE;
                }
            } else {
                for (int s = p-1; s >= 0; s--) mtxfile_free(dsts[s]);
                free(partsizes);
                mtxfiledata_free(
                    &data, src->header.object, src->header.format,
                    src->header.field, src->precision);
                return MTX_ERR_INVALID_MTX_FORMAT;
            }
        } else {
            for (int s = p-1; s >= 0; s--) mtxfile_free(dsts[s]);
            free(partsizes);
            mtxfiledata_free(
                &data, src->header.object, src->header.format,
                src->header.field, src->precision);
            return MTX_ERR_INVALID_MTX_OBJECT;
        }

        err = mtxfile_alloc(
            dsts[p], &src->header, &src->comments,  &dstsize, src->precision);
        if (err) {
            for (int s = p-1; s >= 0; s--) mtxfile_free(dsts[s]);
            free(partsizes);
            mtxfiledata_free(
                &data, src->header.object, src->header.format,
                src->header.field, src->precision);
            return err;
        }

        err = mtxfiledata_copy(
            &dsts[p]->data, &data, dsts[p]->header.object, dsts[p]->header.format,
            dsts[p]->header.field, dsts[p]->precision, partsizes[p], 0, srcoffset);
        if (err) {
            for (int s = p; s >= 0; s--) mtxfile_free(dsts[s]);
            free(partsizes);
            mtxfiledata_free(
                &data, src->header.object, src->header.format,
                src->header.field, src->precision);
            return err;
        }
        srcoffset += partsizes[p];
    }

    free(partsizes);
    mtxfiledata_free(
        &data, src->header.object, src->header.format,
        src->header.field, src->precision);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_join()’ joins together Matrix Market files representing
 * compatible blocks of a partitioned matrix or vector to form a
 * larger matrix or vector.
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
 * of ‘srcs[p*Q+q]->size.num_rows’ for ‘p=0,1,...,P-1’ must be equal
 * to ‘rowpart->size’. Likewise, for each block row ‘p’, the sum of
 * ‘srcs[p*Q+q]->size.num_rows’ for ‘q=0,1,...,Q-1’ must be equal to
 * ‘colpart->size’.
 */
int mtxfile_join(
    struct mtxfile * dst,
    const struct mtxfile * srcs,
    const struct mtxpartition * rowpart,
    const struct mtxpartition * colpart)
{
    int err;
    int num_row_parts = rowpart ? rowpart->num_parts : 1;
    int num_col_parts = colpart ? colpart->num_parts : 1;
    if (num_row_parts <= 0 || num_col_parts <= 0) return MTX_ERR_INVALID_PARTITION;

    /* Check that the blocks are of the same type */
    struct mtxfileheader header;
    err = mtxfileheader_copy(&header, &srcs[0].header);
    if (err) return err;
    enum mtxprecision precision = srcs[0].precision;
    for (int p = 0; p < num_row_parts; p++) {
        for (int q = 0; q < num_col_parts; q++) {
            const struct mtxfile * src = &srcs[p*num_col_parts+q];
            if (src->header.object != header.object)
                return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
            if (src->header.format != header.format)
                return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
            if (src->header.field != header.field)
                return MTX_ERR_INCOMPATIBLE_MTX_FIELD;
            if (src->header.symmetry != header.symmetry)
                return MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY;
            if (src->precision != precision)
                return MTX_ERR_INCOMPATIBLE_PRECISION;
        }
    }

    /* Check that the blocks are compatible in size */
    struct mtxfilesize size;
    size.num_rows = rowpart ? rowpart->size : srcs[0].size.num_rows;
    size.num_columns = colpart ? colpart->size : srcs[0].size.num_columns;
    size.num_nonzeros = srcs[0].size.num_nonzeros == -1 ? -1 : 0;
    for (int q = 0; q < num_col_parts; q++) {
        int64_t num_rows =
            srcs[0*num_col_parts+q].size.num_rows == -1 ? -1 : 0;
        for (int p = 0; p < num_row_parts; p++) {
            const struct mtxfile * src = &srcs[p*num_col_parts+q];
            if (rowpart && rowpart->part_sizes[p] !=
                (src->size.num_rows >= 0 ? src->size.num_rows : 1))
                return MTX_ERR_INCOMPATIBLE_PARTITION;
            if (num_rows >= 0 && src->size.num_rows >= 0) {
                num_rows += src->size.num_rows;
            } else if (num_rows < 0 && src->size.num_rows < 0) {
                /* do nothing */
            } else { return MTX_ERR_INCOMPATIBLE_MTX_SIZE; }
        }
        if (num_rows != size.num_rows)
            return MTX_ERR_INCOMPATIBLE_PARTITION;
    }
    for (int p = 0; p < num_row_parts; p++) {
        int64_t num_columns =
            srcs[p*num_col_parts+0].size.num_columns == -1 ? -1 : 0;
        for (int q = 0; q < num_col_parts; q++) {
            const struct mtxfile * src = &srcs[p*num_col_parts+q];
            if (colpart && colpart->part_sizes[q] !=
                (src->size.num_columns >= 0 ? src->size.num_columns : 1))
                return MTX_ERR_INCOMPATIBLE_PARTITION;
            if (num_columns >= 0 && src->size.num_columns >= 0) {
                num_columns += src->size.num_columns;
            } else if (num_columns < 0 && src->size.num_columns < 0) {
                /* do nothing */
            } else { return MTX_ERR_INCOMPATIBLE_MTX_SIZE; }
        }
        if (num_columns != size.num_columns)
            return MTX_ERR_INCOMPATIBLE_PARTITION;
    }
    for (int p = 0; p < num_row_parts; p++) {
        for (int q = 0; q < num_col_parts; q++) {
            const struct mtxfile * src = &srcs[p*num_col_parts+q];
            if (size.num_nonzeros >= 0 && src->size.num_nonzeros >= 0) {
                size.num_nonzeros += src->size.num_nonzeros;
            } else if (size.num_nonzeros < 0 && src->size.num_nonzeros < 0) {
                /* do nothing */
            } else { return MTX_ERR_INCOMPATIBLE_MTX_SIZE; }
        }
    }

    /* Concatenate comments from each block */
    struct mtxfilecomments comments;
    err = mtxfilecomments_init(&comments);
    if (err) return err;
    for (int p = 0; p < num_row_parts; p++) {
        for (int q = 0; q < num_col_parts; q++) {
            const struct mtxfile * src = &srcs[p*num_col_parts+q];
            err = mtxfilecomments_cat(&comments, &src->comments);
            if (err) {
                mtxfilecomments_free(&comments);
                return err;
            }
        }
    }

    /* Allocate storage for the joined matrix or vector */
    err = mtxfile_alloc(dst, &header, &comments, &size, precision);
    if (err) {
        mtxfilecomments_free(&comments);
        return err;
    }
    mtxfilecomments_free(&comments);

    if (dst->header.format == mtxfile_array) {
        /* If the matrix or vector is in array format, we first obtain
         * the mapping from local, partwise row and column numbers to
         * the global row and column numbers of the joined matrix or
         * vector. Then we sort the matrix or vector in row major
         * order. */

        /* Allocate sorting keys */
        uint64_t * sortkeys = malloc(dst->datasize * sizeof(uint64_t));
        if (!sortkeys) {
            mtxfile_free(dst);
            return MTX_ERR_ERRNO;
        }

        int64_t dstoffset = 0;
        for (int p = 0; p < num_row_parts; p++) {
            for (int q = 0; q < num_col_parts; q++) {
                int r = p*num_col_parts+q;
                const struct mtxfile * src = &srcs[r];
                err = mtxfiledata_copy(
                    &dst->data, &src->data,
                    src->header.object, src->header.format,
                    src->header.field, src->precision,
                    src->datasize, dstoffset, 0);
                if (err) {
                    free(sortkeys);
                    mtxfile_free(dst);
                    return err;
                }

                int * localrowidx = malloc(src->datasize * sizeof(int));
                if (!localrowidx) {
                    free(sortkeys);
                    mtxfile_free(dst);
                    return MTX_ERR_ERRNO;
                }
                int * localcolidx = malloc(src->datasize * sizeof(int));
                if (!localcolidx) {
                    free(localrowidx);
                    free(sortkeys);
                    mtxfile_free(dst);
                    return MTX_ERR_ERRNO;
                }

                err = mtxfiledata_rowcolidx(
                    &src->data, src->header.object, src->header.format,
                    src->header.field, src->precision,
                    src->size.num_rows, src->size.num_columns,
                    0, src->datasize, localrowidx, localcolidx);
                if (err) {
                    free(localcolidx);
                    free(localrowidx);
                    free(sortkeys);
                    mtxfile_free(dst);
                    return err;
                }

                int64_t * globalrowidx =
                    malloc(src->datasize * sizeof(int64_t));
                if (!globalrowidx) {
                    free(localcolidx);
                    free(localrowidx);
                    free(sortkeys);
                    mtxfile_free(dst);
                    return MTX_ERR_ERRNO;
                }
                for (int64_t k = 0; k < src->datasize; k++)
                    globalrowidx[k] = localrowidx[k]-1;
                int64_t * globalcolidx =
                    malloc(src->datasize * sizeof(int64_t));
                if (!globalcolidx) {
                    free(globalrowidx);
                    free(localcolidx);
                    free(localrowidx);
                    free(sortkeys);
                    mtxfile_free(dst);
                    return MTX_ERR_ERRNO;
                }
                for (int64_t k = 0; k < src->datasize; k++)
                    globalcolidx[k] = localcolidx[k]-1;

                free(localcolidx);
                free(localrowidx);

                if (rowpart) {
                    err = mtxpartition_globalidx(
                        rowpart, p, src->datasize,
                        globalrowidx, globalrowidx);
                    if (err) {
                        free(globalcolidx);
                        free(globalrowidx);
                        free(sortkeys);
                        mtxfile_free(dst);
                        return err;
                    }
                }
                if (colpart) {
                    err = mtxpartition_globalidx(
                        colpart, q, src->datasize,
                        globalcolidx, globalcolidx);
                    if (err) {
                        free(globalcolidx);
                        free(globalrowidx);
                        free(sortkeys);
                        mtxfile_free(dst);
                        return err;
                    }
                }

                for (int64_t k = 0; k < src->datasize; k++) {
                    sortkeys[dstoffset+k] =
                        globalrowidx[k]
                        * (dst->size.num_columns >= 0 ? dst->size.num_columns : 1)
                        + globalcolidx[k];
                }

                free(globalcolidx);
                free(globalrowidx);
                dstoffset += src->datasize;
            }
        }

        err = mtxfiledata_sort_keys(
            &dst->data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision,
            dst->size.num_rows, dst->size.num_columns,
            dst->datasize, sortkeys, NULL);
        if (err) {
            free(sortkeys);
            mtxfile_free(dst);
            return err;
        }
        free(sortkeys);

    } else if (dst->header.format == mtxfile_coordinate) {
        /* If the matrix or vector is in coordinate format, we use
         * mtxfile_reorder to convert from local, partwise numbering
         * of rows and columns to the global numbering. Thereafter, we
         * simply concatenate the nonzeros from each block. */

        int64_t dstoffset = 0;
        for (int p = 0; p < num_row_parts; p++) {
            for (int q = 0; q < num_col_parts; q++) {
                int r = p*num_col_parts+q;
                const struct mtxfile * src = &srcs[r];

                err = mtxfiledata_copy(
                    &dst->data, &src->data,
                    src->header.object, src->header.format,
                    src->header.field, src->precision,
                    src->datasize, dstoffset, 0);
                if (err) {
                    mtxfile_free(dst);
                    return err;
                }

                /* Find row and column permutations needed to convert
                 * from local, partwise to global numbering */
                int64_t * rowperm64 = NULL;
                if (rowpart && src->size.num_rows >= 0) {
                    rowperm64 = malloc(src->size.num_rows * sizeof(int64_t));
                    if (!rowperm64) {
                        mtxfile_free(dst);
                        return MTX_ERR_ERRNO;
                    }
                    for (int i = 0; i < src->size.num_rows; i++)
                        rowperm64[i] = i;
                    err = mtxpartition_globalidx(
                        rowpart, p, src->size.num_rows, rowperm64, rowperm64);
                    if (err) {
                        free(rowperm64);
                        mtxfile_free(dst);
                        return err;
                    }
                }
                int64_t * colperm64 = NULL;
                if (colpart && src->size.num_columns >= 0) {
                    colperm64 = malloc(src->size.num_columns * sizeof(int64_t));
                    if (!colperm64) {
                        if (rowpart && src->size.num_rows >= 0) free(rowperm64);
                        mtxfile_free(dst);
                        return MTX_ERR_ERRNO;
                    }
                    for (int j = 0; j < src->size.num_columns; j++)
                        colperm64[j] = j;
                    err = mtxpartition_globalidx(
                        colpart, q, src->size.num_columns, colperm64, colperm64);
                    if (err) {
                        free(colperm64);
                        if (rowpart && src->size.num_rows >= 0) free(rowperm64);
                        mtxfile_free(dst);
                        return err;
                    }
                }

                int * rowperm = (int *) rowperm64;
                if (rowperm) {
                    for (int i = 0; i < src->size.num_rows; i++)
                        rowperm[i] = rowperm64[i]+1;
                }
                int * colperm = (int *) colperm64;
                if (colperm) {
                    for (int i = 0; i < src->size.num_columns; i++)
                        colperm[i] = colperm64[i]+1;
                }

                /* Convert local, partwise row and column numbers to
                 * global row and column numbers. */
                err = mtxfiledata_reorder(
                    &dst->data, dst->header.object, dst->header.format,
                    dst->header.field, dst->precision,
                    src->datasize, dstoffset,
                    dst->size.num_rows, rowperm,
                    dst->size.num_columns, colperm);
                if (err) {
                    if (colpart && src->size.num_columns >= 0) free(colperm64);
                    if (rowpart && src->size.num_rows >= 0) free(rowperm64);
                    mtxfile_free(dst);
                    return err;
                }

                if (colpart && src->size.num_columns >= 0) free(colperm64);
                if (rowpart && src->size.num_rows >= 0) free(rowperm64);
                dstoffset += src->datasize;
            }
        }
    } else {
        mtxfile_free(dst);
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/*
 * Reordering
 */

/**
 * ‘mtxfile_permute()’ reorders the rows of a vector or the rows and
 *  columns of a matrix in Matrix Market format based on given row and
 *  column permutations.
 *
 * The array ‘row_permutation’ should be a permutation of the integers
 * ‘1,2,...,M’, where ‘M’ is the number of rows in the matrix or
 * vector.  If the Matrix Market file represents a matrix, then the
 * array ‘column_permutation’ should be a permutation of the integers
 * ‘1,2,...,N’, where ‘N’ is the number of columns in the matrix.  The
 * elements belonging to row ‘i’ and column ‘j’ in the permuted matrix
 * are then equal to the elements in row ‘row_permutation[i-1]’ and
 * column ‘column_permutation[j-1]’ in the original matrix, for
 * ‘i=1,2,...,M’ and ‘j=1,2,...,N’.
 */
int mtxfile_permute(
    struct mtxfile * mtxfile,
    const int * row_permutation,
    const int * column_permutation)
{
    return mtxfiledata_reorder(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, mtxfile->datasize, 0,
        mtxfile->size.num_rows, row_permutation,
        mtxfile->size.num_columns, column_permutation);
}

/**
 * ‘mtxfileordering_str()’ is a string representing the ordering
 * of a matrix in Matix Market format.
 */
const char * mtxfileordering_str(
    enum mtxfileordering ordering)
{
    switch (ordering) {
    case mtxfile_default_order: return "default";
    case mtxfile_custom_order: return "custom";
    case mtxfile_rcm: return "rcm";
    default: return mtxstrerror(MTX_ERR_INVALID_ORDERING);
    }
}

/**
 * ‘mtxfileordering_parse()’ parses a string corresponding to a value
 *  of the enum type ‘mtxfileordering’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxfileordering_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘ordering’ is set according to the parsed string and ‘bytes_read’
 * is set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfileordering_parse(
    enum mtxfileordering * ordering,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("default", t, strlen("default")) == 0) {
        t += strlen("default");
        *ordering = mtxfile_default_order;
    } else if (strncmp("custom", t, strlen("custom")) == 0) {
        t += strlen("custom");
        *ordering = mtxfile_custom_order;
    } else if (strncmp("rcm", t, strlen("rcm")) == 0) {
        t += strlen("rcm");
        *ordering = mtxfile_rcm;
    } else {
        return MTX_ERR_INVALID_ORDERING;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_ORDERING;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_reorder_rcm()’ reorders the rows of a sparse matrix
 * according to the Reverse Cuthill-McKee algorithm.
 *
 * For a square matrix, the Cuthill-McKee algorithm is carried out on
 * the adjacency matrix of the symmetrisation ‘A+A'’, where ‘A'’
 * denotes the transpose of ‘A’.  For a rectangular matrix, the
 * Cuthill-McKee algorithm is carried out on a bipartite graph formed
 * by the matrix rows and columns.  The adjacency matrix ‘B’ of the
 * bipartite graph is square and symmetric and takes the form of a
 * 2-by-2 block matrix where ‘A’ is placed in the upper right corner
 * and ‘A'’ is placed in the lower left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * ‘starting_vertex’ is a pointer to an integer which can be used to
 * designate a starting vertex for the Cuthill-McKee algorithm.
 * Alternatively, setting the starting_vertex to zero causes a
 * starting vertex to be chosen automatically by selecting a
 * pseudo-peripheral vertex.
 *
 * In the case of a square matrix, the starting vertex must be in the
 * range ‘[1,M]’, where ‘M’ is the number of rows (and columns) of the
 * matrix.  Otherwise, if the matrix is rectangular, a starting vertex
 * in the range ‘[1,M]’ selects a vertex corresponding to a row of the
 * matrix, whereas a starting vertex in the range ‘[M+1,M+N]’, where
 * ‘N’ is the number of matrix columns, selects a vertex corresponding
 * to a column of the matrix.
 *
 * The reordering is symmetric if the matrix is square, and
 * unsymmetric otherwise.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered according to the
 * Reverse Cuthill-McKee algorithm. If ‘rowperm’ is not ‘NULL’, then
 * it must point to an array that is large enough to hold one ‘int’
 * for each row of the matrix. In this case, the array is used to
 * store the permutation for reordering the matrix rows. Similarly,
 * ‘colperm’ is used to store the permutation for reordering the
 * matrix columns.
 */
int mtxfile_reorder_rcm(
    struct mtxfile * mtxfile,
    int * rowperm,
    int * colperm,
    bool permute,
    bool * symmetric,
    int * starting_vertex)
{
    int err;
    int64_t num_rows = mtxfile->size.num_rows;
    int64_t num_columns = mtxfile->size.num_columns;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;
    bool square = num_rows == num_columns;
    int num_vertices = square ? num_rows : num_rows + num_columns;

    if (symmetric) *symmetric = square;
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxfile->header.format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (starting_vertex && (*starting_vertex < 0 || *starting_vertex > num_vertices))
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    /* 1. Obtain row pointers and column indices */
    int64_t * rowptr = malloc(
        (num_rows+1)*sizeof(int64_t) + num_nonzeros*sizeof(int));
    if (!rowptr)
        return MTX_ERR_ERRNO;
    int * colidx = (int *) (&rowptr[num_rows+1]);
    err = mtxfiledata_rowptr(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_rows, num_nonzeros,
        rowptr, colidx, NULL);
    if (err) {
        free(rowptr);
        return err;
    }

    /* 2. Obtain column pointers and row indices, which is equivalent
     * to row pointers and column indices of the transposed matrix. */
    int64_t * colptr = malloc(
        (num_columns+1)*sizeof(int64_t) + num_nonzeros*sizeof(int));
    if (!colptr) {
        free(rowptr);
        return MTX_ERR_ERRNO;
    }
    int * rowidx = (int *) (&colptr[num_columns+1]);
    err = mtxfiledata_colptr(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_columns, num_nonzeros,
        colptr, rowidx, NULL);
    if (err) {
        free(colptr);
        free(rowptr);
        return err;
    }

    /* 3. Allocate storage for vertex ordering */
    int * vertex_order = malloc(num_vertices * sizeof(int));
    if (!vertex_order) {
        free(colptr);
        free(rowptr);
        return MTX_ERR_ERRNO;
    }

    /* 4. Compute the Cuthill-McKee ordering */
    err = cuthill_mckee(
        num_rows, num_columns, rowptr, colidx, colptr, rowidx,
        starting_vertex, num_vertices, vertex_order);
    if (err) {
        free(vertex_order);
        free(colptr);
        free(rowptr);
        return err;
    }

    free(colptr);
    free(rowptr);

    /* Add one to shift from 0-based to 1-based indexing. */
    for (int i = 0; i < num_vertices; i++)
        vertex_order[i]++;

    /* 5. Reverse the ordering. */
    for (int i = 0; i < num_vertices/2; i++) {
        int tmp = vertex_order[i];
        vertex_order[i] = vertex_order[num_vertices-i-1];
        vertex_order[num_vertices-i-1] = tmp;
    }

    bool alloc_rowperm = !rowperm;
    if (alloc_rowperm) {
        rowperm = malloc(num_rows * sizeof(int));
        if (!rowperm) {
            free(vertex_order);
            return MTX_ERR_ERRNO;
        }
    }
    bool alloc_colperm = !square && !colperm;
    if (alloc_colperm) {
        colperm = malloc(num_columns * sizeof(int));
        if (!colperm) {
            if (alloc_rowperm)
                free(rowperm);
            free(vertex_order);
            return MTX_ERR_ERRNO;
        }
    }

    if (square) {
        for (int i = 0; i < num_vertices; i++)
            rowperm[vertex_order[i]-1] = i+1;
        if (colperm) {
            for (int i = 0; i < num_vertices; i++)
                colperm[i] = rowperm[i];
        }
    } else {
        int * row_order = malloc((num_rows + num_columns) * sizeof(int));
        if (!row_order) {
            if (alloc_colperm)
                free(colperm);
            if (alloc_rowperm)
                free(rowperm);
            free(vertex_order);
            return MTX_ERR_ERRNO;
        }
        int * col_order = &row_order[num_rows];

        int k = 0;
        int l = 0;
        for (int i = 0; i < num_vertices; i++) {
            if (vertex_order[i] <= num_rows) {
                row_order[k] = vertex_order[i];
                k++;
            } else {
                col_order[l] = vertex_order[i]-num_rows;
                l++;
            }
        }

        for (int i = 0; i < num_rows; i++)
            rowperm[row_order[i]-1] = i+1;
        for (int i = 0; i < num_columns; i++)
            colperm[col_order[i]-1] = i+1;
        free(row_order);
    }

    /* 7. Permute the matrix. */
    if (permute) {
        err = mtxfile_permute(
            mtxfile, rowperm, square ? rowperm : colperm);
        if (err) {
            if (alloc_colperm)
                free(colperm);
            if (alloc_rowperm)
                free(rowperm);
            free(vertex_order);
            return err;
        }
    }

    if (alloc_colperm)
        free(colperm);
    if (alloc_rowperm)
        free(rowperm);
    free(vertex_order);
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_reorder()’ reorders the rows and columns of a matrix
 * according to the specified algorithm.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered according to the
 * specified method. If ‘rowperm’ is not ‘NULL’, then it must point to
 * an array that is large enough to hold one ‘int’ for each row of the
 * matrix. In this case, the array is used to store the permutation
 * for reordering the matrix rows. Similarly, ‘colperm’ is used to
 * store the permutation for reordering the matrix columns.
 *
 * If ‘symmetric’ is not ‘NULL’, then it is used to return whether or
 * not the reordering is symmetric. That is, if the value returned in
 * ‘symmetric’ is ‘true’ then ‘rowperm’ and ‘colperm’ are identical,
 * and only one of them is needed.
 */
int mtxfile_reorder(
    struct mtxfile * mtxfile,
    enum mtxfileordering ordering,
    int * rowperm,
    int * colperm,
    bool permute,
    bool * symmetric,
    int * rcm_starting_vertex)
{
    /* rectangular matrices always yield unsymmetric orderings */
    int64_t num_rows = mtxfile->size.num_rows;
    int64_t num_columns = mtxfile->size.num_columns;
    bool square = num_rows == num_columns;

    if (ordering == mtxfile_default_order) {
        if (symmetric) *symmetric = !square;
        if (rowperm) {
            for (int i = 0; i < num_rows; i++)
                rowperm[i] = i+1;
        }
        if (colperm) {
            for (int j = 0; j < num_columns; j++)
                colperm[j] = j+1;
        }
        return MTX_SUCCESS;
    } else if (ordering == mtxfile_custom_order) {
        if (symmetric && square && rowperm == colperm) {
            *symmetric = true;
        } else if (symmetric && square && rowperm != NULL && colperm != NULL) {
            *symmetric = true;
            for (int i = 0; i < num_rows; i++) {
                if (rowperm[i] != colperm[i]) {
                    *symmetric = false;
                    break;
                }
            }
        } else if (symmetric) {
            *symmetric = false;
        }
        return permute ? mtxfile_permute(mtxfile, rowperm, colperm) : MTX_SUCCESS;
    } else if (ordering == mtxfile_rcm) {
        return mtxfile_reorder_rcm(
            mtxfile, rowperm, colperm, permute,
            symmetric, rcm_starting_vertex);
    } else {
        return MTX_ERR_INVALID_ORDERING;
    }
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxfile_send()’ sends a Matrix Market file to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxfile_recv()’.
 */
int mtxfile_send(
    const struct mtxfile * mtxfile,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxfileheader_send(&mtxfile->header, dest, tag, comm, disterr);
    if (err) return err;
    err = mtxfilecomments_send(&mtxfile->comments, dest, tag, comm, disterr);
    if (err) return err;
    err = mtxfilesize_send(&mtxfile->size, dest, tag, comm, disterr);
    if (err) return err;
    disterr->mpierrcode = MPI_Send(
        &mtxfile->precision, 1, MPI_INT, dest, tag, comm);
    if (disterr->mpierrcode) return MTX_ERR_MPI;
    disterr->mpierrcode = MPI_Send(
        &mtxfile->datasize, 1, MPI_INT64_T, dest, tag, comm);
    if (disterr->mpierrcode) return MTX_ERR_MPI;
    return mtxfiledata_send(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        mtxfile->datasize, 0,
        dest, tag, comm, disterr);
}

/**
 * ‘mtxfile_recv()’ receives a Matrix Market file from another MPI
 * process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxfile_send()’.
 */
int mtxfile_recv(
    struct mtxfile * mtxfile,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxfileheader_recv(&mtxfile->header, source, tag, comm, disterr);
    if (err) return err;
    err = mtxfilecomments_recv(&mtxfile->comments, source, tag, comm, disterr);
    if (err) return err;
    err = mtxfilesize_recv(&mtxfile->size, source, tag, comm, disterr);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    disterr->mpierrcode = MPI_Recv(
        &mtxfile->precision, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->mpierrcode) {
        mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI;
    }
    disterr->mpierrcode = MPI_Recv(
        &mtxfile->datasize, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    if (disterr->mpierrcode) {
        mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI;
    }

    err = mtxfiledata_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision, mtxfile->datasize);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }

    err = mtxfiledata_recv(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        mtxfile->datasize, 0,
        source, tag, comm, disterr);
    if (err) {
        mtxfiledata_free(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision);
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_bcast()’ broadcasts a Matrix Market file from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_bcast()’.
 */
int mtxfile_bcast(
    struct mtxfile * mtxfile,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    err = mtxfileheader_bcast(&mtxfile->header, root, comm, disterr);
    if (err) return err;
    err = mtxfilecomments_bcast(&mtxfile->comments, root, comm, disterr);
    if (err) return err;
    err = mtxfilesize_bcast(&mtxfile->size, root, comm, disterr);
    if (err) {
        if (rank != root)
            mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    disterr->mpierrcode = MPI_Bcast(
        &mtxfile->precision, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank != root)
            mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Bcast(
        &mtxfile->datasize, 1, MPI_INT64_T, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank != root)
            mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rank != root) {
        err = mtxfiledata_alloc(
            &mtxfile->data,
            mtxfile->header.object,
            mtxfile->header.format,
            mtxfile->header.field,
            mtxfile->precision, mtxfile->datasize);
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        if (rank != root)
            mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_bcast(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        mtxfile->datasize, 0,
        root, comm, disterr);
    if (err) {
        if (rank != root) {
            mtxfiledata_free(
                &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
                mtxfile->header.field, mtxfile->precision);
            mtxfilecomments_free(&mtxfile->comments);
        }
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_gather()’ gathers Matrix Market files onto an MPI root
 * process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gather()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_gather()’.
 */
int mtxfile_gather(
    const struct mtxfile * sendmtxfile,
    struct mtxfile * recvmtxfiles,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    for (int p = 0; p < comm_size; p++) {
        /* Send to the root process */
        err = (rank != root && rank == p)
            ? mtxfile_send(sendmtxfile, root, 0, comm, disterr)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                for (int q = p-1; q >= 0; q--)
                    mtxfile_free(&recvmtxfiles[q]);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Receive on the root process */
        err = (rank == root && p != root)
            ? mtxfile_recv(&recvmtxfiles[p], p, 0, comm, disterr)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                for (int q = p-1; q >= 0; q--)
                    mtxfile_free(&recvmtxfiles[q]);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Perform a copy on the root process */
        err = (rank == root && p == root)
            ? mtxfile_init_copy(&recvmtxfiles[p], sendmtxfile) : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank == root) {
                for (int q = p-1; q >= 0; q--)
                    mtxfile_free(&recvmtxfiles[q]);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_allgather()’ gathers Matrix Market files onto every MPI
 * process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Allgather()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxfile_allgather()’.
 */
int mtxfile_allgather(
    const struct mtxfile * sendmtxfile,
    struct mtxfile * recvmtxfiles,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int p = 0; p < comm_size; p++) {
        err = mtxfile_gather(sendmtxfile, recvmtxfiles, p, comm, disterr);
        if (err)
            return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_scatter()’ scatters Matrix Market files from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatter()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_scatter()’.
 */
int mtxfile_scatter(
    const struct mtxfile * sendmtxfiles,
    struct mtxfile * recvmtxfile,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    for (int p = 0; p < comm_size; p++) {
        if (rank == root && p != root) {
            /* Send from the root process */
            err = mtxfile_send(&sendmtxfiles[p], p, 0, comm, disterr);
        } else if (rank != root && rank == p) {
            /* Receive from the root process */
            err = mtxfile_recv(recvmtxfile, root, 0, comm, disterr);
        } else if (rank == root && p == root) {
            /* Perform a copy on the root process */
            err = mtxfile_init_copy(recvmtxfile, &sendmtxfiles[p]);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxdisterror_allreduce(disterr, err)) {
            if (rank < p)
                mtxfile_free(recvmtxfile);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_alltoall()’ performs an all-to-all exchange of Matrix
 * Market files between MPI process in a communicator.
 *
 * This is analogous to ‘MPI_Alltoall()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_alltoall()’.
 */
int mtxfile_alltoall(
    const struct mtxfile * sendmtxfiles,
    struct mtxfile * recvmtxfiles,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    for (int p = 0; p < comm_size; p++) {
        err = mtxfile_scatter(sendmtxfiles, &recvmtxfiles[p], p, comm, disterr);
        if (err) {
            for (int q = p-1; q >= 0; q--)
                mtxfile_free(&recvmtxfiles[q]);
            return err;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_scatterv()’ scatters a Matrix Market file from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to ‘mtxfile_scatterv()’.
 */
int mtxfile_scatterv(
    const struct mtxfile * sendmtxfile,
    const int * sendcounts,
    const int * displs,
    struct mtxfile * recvmtxfile,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    int comm_size;
    disterr->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int rank;
    disterr->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;

    err = (rank == root)
        ? mtxfileheader_copy(&recvmtxfile->header, &sendmtxfile->header)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfileheader_bcast(&recvmtxfile->header, root, comm, disterr);
    if (err)
        return err;

    err = (rank == root)
        ? mtxfilecomments_init(&recvmtxfile->comments)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == root)
        ? mtxfilecomments_copy(&recvmtxfile->comments, &sendmtxfile->comments)
        : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfilecomments_bcast(&recvmtxfile->comments, root, comm, disterr);
    if (err) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return err;
    }

    err = mtxfilesize_scatterv(
        &sendmtxfile->size, &recvmtxfile->size,
        recvmtxfile->header.object, recvmtxfile->header.format,
        recvcount, root, comm, disterr);
    if (err) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return err;
    }

    recvmtxfile->precision = sendmtxfile->precision;
    disterr->mpierrcode = MPI_Bcast(
        &recvmtxfile->precision, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfilesize_num_data_lines(
        &recvmtxfile->size, recvmtxfile->header.symmetry, &recvmtxfile->datasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = recvcount != recvmtxfile->datasize ? MTX_ERR_INDEX_OUT_OF_BOUNDS : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_alloc(
        &recvmtxfile->data,
        recvmtxfile->header.object,
        recvmtxfile->header.format,
        recvmtxfile->header.field,
        recvmtxfile->precision,
        recvmtxfile->datasize);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_scatterv(
        &sendmtxfile->data,
        recvmtxfile->header.object, recvmtxfile->header.format,
        recvmtxfile->header.field, recvmtxfile->precision,
        0, sendcounts, displs,
        &recvmtxfile->data, 0, recvmtxfile->datasize,
        root, comm, disterr);
    if (err) {
        mtxfiledata_free(
            &recvmtxfile->data, recvmtxfile->header.object, recvmtxfile->header.format,
            recvmtxfile->header.field, recvmtxfile->precision);
        mtxfilecomments_free(&recvmtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}
#endif
