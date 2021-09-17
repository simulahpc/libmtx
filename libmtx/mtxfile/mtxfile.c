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
 * Last modified: 2021-09-01
 *
 * Matrix Market files.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/coordinate.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>

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
 * `mtxfile_free()' frees storage allocated for a Matrix Market file.
 */
void mtxfile_free(
    struct mtxfile * mtxfile)
{
    mtxfile_data_free(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision);
    mtxfile_comments_free(&mtxfile->comments);
}

/**
 * `mtxfile_copy()' copies a Matrix Market file.
 */
int mtxfile_copy(
    struct mtxfile * dst,
    const struct mtxfile * src)
{
    int err;
    err = mtxfile_header_copy(&dst->header, &src->header);
    if (err)
        return err;
    err = mtxfile_comments_copy(&dst->comments, &src->comments);
    if (err)
        return err;
    err = mtxfile_size_copy(&dst->size, &src->size);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &src->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    err = mtxfile_data_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        num_data_lines, 0, 0);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_cat()' concatenates two Matrix Market files.
 *
 * The files must have identical header lines. Furthermore, for
 * matrices in array format, both matrices must have the same number
 * of columns, since entire rows are concatenated.  For matrices or
 * vectors in coordinate format, the number of rows and columns must
 * be the same.
 */
int mtxfile_cat(
    struct mtxfile * dst,
    const struct mtxfile * src)
{
    int err;
    if (dst->header.object != src->header.object)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (dst->header.format != src->header.format)
        return MTX_ERR_INVALID_MTX_FORMAT;
    if (dst->header.field != src->header.field)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (dst->header.symmetry != src->header.symmetry)
        return MTX_ERR_INVALID_MTX_SYMMETRY;

    err = mtxfile_comments_cat(&dst->comments, &src->comments);
    if (err)
        return err;

    int64_t num_data_lines_dst;
    err = mtxfile_size_num_data_lines(&dst->size, &num_data_lines_dst);
    if (err)
        return err;
    int64_t num_data_lines_src;
    err = mtxfile_size_num_data_lines(&src->size, &num_data_lines_src);
    if (err)
        return err;

    err = mtxfile_size_cat(
        &dst->size, &src->size, dst->header.object, dst->header.format);
    if (err)
        return err;

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&dst->size, &num_data_lines);
    if (err)
        return err;
    if (num_data_lines_dst + num_data_lines_src != num_data_lines)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    union mtxfile_data data;
    err = mtxfile_data_alloc(
        &data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_data_lines);
    if (err)
        return err;

    err = mtxfile_data_copy(
        &data, &dst->data,
        dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        num_data_lines_dst, 0, 0);
    if (err) {
        mtxfile_data_free(
            &data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision);
        return err;
    }

    err = mtxfile_data_copy(
        &data, &src->data,
        dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        num_data_lines_src, num_data_lines_dst, 0);
    if (err) {
        mtxfile_data_free(
            &data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision);
        return err;
    }

    union mtxfile_data olddata = dst->data;
    dst->data = data;
    mtxfile_data_free(
        &olddata, dst->header.object, dst->header.format,
        dst->header.field, dst->precision);
    return MTX_SUCCESS;
}

/*
 * Matrix array formats
 */

/**
 * `mtxfile_alloc_matrix_array()' allocates a matrix in array format.
 */
int mtxfile_alloc_matrix_array(
    struct mtxfile * mtxfile,
    enum mtxfile_field field,
    enum mtxfile_symmetry symmetry,
    enum mtx_precision precision,
    int num_rows,
    int num_columns)
{
    int err;
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
    mtxfile_comments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = num_columns;
    mtxfile->size.num_nonzeros = -1;
    mtxfile->precision = precision;

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err)
        return err;

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        num_data_lines);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_array_real_single()' allocates and initialises
 * a matrix in array format with real, single precision coefficients.
 */
int mtxfile_init_matrix_array_real_single(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const float * data);

/**
 * `mtxfile_init_matrix_array_real_double()' allocates and initialises
 * a matrix in array format with real, double precision coefficients.
 */
int mtxfile_init_matrix_array_real_double(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const double * data)
{
    int err;
    err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_real, symmetry, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_free(mtxfile);
        return err;
    }
    memcpy(mtxfile->data.array_real_double, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_array_complex_single()' allocates and
 * initialises a matrix in array format with complex, single precision
 * coefficients.
 */
int mtxfile_init_matrix_array_complex_single(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const float (* data)[2]);

/**
 * `mtxfile_init_matrix_array_complex_double()' allocates and
 * initialises a matrix in array format with complex, double precision
 * coefficients.
 */
int mtxfile_init_matrix_array_complex_double(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const double (* data)[2]);

/**
 * `mtxfile_init_matrix_array_integer_single()' allocates and
 * initialises a matrix in array format with integer, single precision
 * coefficients.
 */
int mtxfile_init_matrix_array_integer_single(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const int32_t * data);

/**
 * `mtxfile_init_matrix_array_integer_double()' allocates and
 * initialises a matrix in array format with integer, double precision
 * coefficients.
 */
int mtxfile_init_matrix_array_integer_double(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    const int64_t * data);

/*
 * Vector array formats
 */

/**
 * `mtxfile_alloc_vector_array()' allocates a vector in array format.
 */
int mtxfile_alloc_vector_array(
    struct mtxfile * mtxfile,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows)
{
    int err;
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
    mtxfile_comments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = -1;
    mtxfile->size.num_nonzeros = -1;
    mtxfile->precision = precision;
    err = mtxfile_data_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_rows);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_array_real_single()' allocates and initialises
 * a vector in array format with real, single precision coefficients.
 */
int mtxfile_init_vector_array_real_single(
    struct mtxfile * mtxfile,
    int num_rows,
    const float * data)
{
    int err = mtxfile_alloc_vector_array(mtxfile, mtxfile_real, mtx_single, num_rows);
    if (err)
        return err;
    memcpy(mtxfile->data.array_real_single, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_array_real_double()' allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxfile_init_vector_array_real_double(
    struct mtxfile * mtxfile,
    int num_rows,
    const double * data)
{
    int err = mtxfile_alloc_vector_array(mtxfile, mtxfile_real, mtx_double, num_rows);
    if (err)
        return err;
    memcpy(mtxfile->data.array_real_double, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_array_complex_single()' allocates and
 * initialises a vector in array format with complex, single precision
 * coefficients.
 */
int mtxfile_init_vector_array_complex_single(
    struct mtxfile * mtxfile,
    int num_rows,
    const float (* data)[2])
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_complex, mtx_single, num_rows);
    if (err)
        return err;
    memcpy(mtxfile->data.array_complex_single, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_array_complex_double()' allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxfile_init_vector_array_complex_double(
    struct mtxfile * mtxfile,
    int num_rows,
    const double (* data)[2])
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_complex, mtx_double, num_rows);
    if (err)
        return err;
    memcpy(mtxfile->data.array_complex_double, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_array_integer_single()' allocates and
 * initialises a vector in array format with integer, single precision
 * coefficients.
 */
int mtxfile_init_vector_array_integer_single(
    struct mtxfile * mtxfile,
    int num_rows,
    const int32_t * data);

/**
 * `mtxfile_init_vector_array_integer_double()' allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_array_integer_double(
    struct mtxfile * mtxfile,
    int num_rows,
    const int64_t * data);

/*
 * Matrix coordinate formats
 */

/**
 * `mtxfile_alloc_matrix_coordinate()' allocates a matrix in
 * coordinate format.
 */
int mtxfile_alloc_matrix_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfile_field field,
    enum mtxfile_symmetry symmetry,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros)
{
    int err;
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
    mtxfile->header.format = mtxfile_coordinate;
    mtxfile->header.field = field;
    mtxfile->header.symmetry = symmetry;
    mtxfile_comments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = num_columns;
    mtxfile->size.num_nonzeros = num_nonzeros;
    mtxfile->precision = precision;

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        num_nonzeros);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_real_single()' allocates and initialises
 * a matrix in coordinate format with real, single precision coefficients.
 */
int mtxfile_init_matrix_coordinate_real_single(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_single * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_real, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_real_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_real_double()' allocates and initialises
 * a matrix in coordinate format with real, double precision coefficients.
 */
int mtxfile_init_matrix_coordinate_real_double(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_real_double * data)
{
    int err;
    err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_real, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_real_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_complex_single()' allocates and
 * initialises a matrix in coordinate format with complex, single precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_complex_single(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_single * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_complex, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_complex_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_complex_double()' allocates and
 * initialises a matrix in coordinate format with complex, double precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_complex_double(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_complex_double * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_complex, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_complex_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_integer_single()' allocates and
 * initialises a matrix in coordinate format with integer, single precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_integer_single(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_single * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_integer, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_integer_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_integer_double()' allocates and
 * initialises a matrix in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_integer_double(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_integer_double * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_integer, symmetry, mtx_double,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_integer_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_matrix_coordinate_pattern()' allocates and
 * initialises a matrix in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_matrix_coordinate_pattern(
    struct mtxfile * mtxfile,
    enum mtxfile_symmetry symmetry,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const struct mtxfile_matrix_coordinate_pattern * data)
{
    int err = mtxfile_alloc_matrix_coordinate(
        mtxfile, mtxfile_pattern, symmetry, mtx_single,
        num_rows, num_columns, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.matrix_coordinate_pattern,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * Vector coordinate formats
 */

/**
 * `mtxfile_alloc_vector_coordinate()' allocates a vector in
 * coordinate format.
 */
int mtxfile_alloc_vector_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int64_t num_nonzeros)
{
    int err;
    if (field != mtxfile_real &&
        field != mtxfile_complex &&
        field != mtxfile_integer)
        return MTX_ERR_INVALID_MTX_FIELD;
    if (precision != mtx_single &&
        precision != mtx_double)
        return MTX_ERR_INVALID_PRECISION;

    mtxfile->header.object = mtxfile_vector;
    mtxfile->header.format = mtxfile_coordinate;
    mtxfile->header.field = field;
    mtxfile->header.symmetry = mtxfile_general;
    mtxfile_comments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = -1;
    mtxfile->size.num_nonzeros = num_nonzeros;
    mtxfile->precision = precision;

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        num_nonzeros);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_real_single()' allocates and initialises
 * a vector in coordinate format with real, single precision coefficients.
 */
int mtxfile_init_vector_coordinate_real_single(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_real, mtx_single, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_real_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_real_double()' allocates and initialises
 * a vector in coordinate format with real, double precision coefficients.
 */
int mtxfile_init_vector_coordinate_real_double(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_real, mtx_double, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_real_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_complex_single()' allocates and
 * initialises a vector in coordinate format with complex, single precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_complex_single(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_complex, mtx_single, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_complex_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_complex_double()' allocates and
 * initialises a vector in coordinate format with complex, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_complex_double(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_complex, mtx_double, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_complex_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_integer_single()' allocates and
 * initialises a vector in coordinate format with integer, single precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_integer_single(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_integer, mtx_single, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_integer_single,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_integer_double()' allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_integer_double(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_integer, mtx_double, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_integer_double,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_coordinate_pattern()' allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_pattern(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data)
{
    int err = mtxfile_alloc_vector_coordinate(
        mtxfile, mtxfile_pattern, mtx_single, num_rows, num_nonzeros);
    if (err)
        return err;
    memcpy(mtxfile->data.vector_coordinate_pattern,
           data, num_nonzeros * sizeof(*data));
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * `mtxfile_read()' reads a Matrix Market file from the given path.
 * The file may optionally be compressed by gzip.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_read(
    struct mtxfile * mtxfile,
    enum mtx_precision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read)
{
    int err;
    *lines_read = -1;
    *bytes_read = 0;

    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            f = stdin;
        } else if ((f = fopen(path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        *lines_read = 0;
        err = mtxfile_fread(
            mtxfile, precision, f, lines_read, bytes_read, 0, NULL);
        if (err)
            return err;
        fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            f = gzdopen(STDIN_FILENO, "r");
        } else if ((f = gzopen(path, "r")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        *lines_read = 0;
        err = mtxfile_gzread(
            mtxfile, precision, f, lines_read, bytes_read, 0, NULL);
        if (err)
            return err;
        gzclose(f);
#else
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_fread()' reads a Matrix Market file from a stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_fread(
    struct mtxfile * mtxfile,
    enum mtx_precision precision,
    FILE * f,
    int * lines_read,
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

    err = mtxfile_fread_header(
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

    err = mtxfile_fread_size(
        &mtxfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object, mtxfile->header.format);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_fread_data(
        &mtxfile->data, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision,
        mtxfile->size.num_rows,
        mtxfile->size.num_columns,
        num_data_lines);
    if (err) {
        mtxfile_data_free(
            &mtxfile->data, mtxfile->header.object,
            mtxfile->header.format, mtxfile->header.field,
            precision);
        mtxfile_comments_free(&mtxfile->comments);
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
 * `mtxfile_gzread()' reads a Matrix Market file from a
 * gzip-compressed stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxfile_gzread(
    struct mtxfile * mtxfile,
    enum mtx_precision precision,
    gzFile f,
    int * lines_read,
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

    err = mtxfile_gzread_header(
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

    err = mtxfile_gzread_size(
        &mtxfile->size, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object, mtxfile->header.format);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    err = mtxfile_gzread_data(
        &mtxfile->data, f, lines_read, bytes_read, line_max, linebuf,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        precision,
        mtxfile->size.num_rows,
        mtxfile->size.num_columns,
        num_data_lines);
    if (err) {
        mtxfile_data_free(
            &mtxfile->data, mtxfile->header.object,
            mtxfile->header.format, mtxfile->header.field,
            precision);
        mtxfile_comments_free(&mtxfile->comments);
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
 * `mtxfile_write()' writes a Matrix Market file to the given path.
 * Market format. The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtxfile_write(
    const struct mtxfile * mtxfile,
    const char * path,
    bool gzip,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    *bytes_written = 0;

    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            f = stdout;
        } else if ((f = fopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        err = mtxfile_fwrite(mtxfile, f, format, bytes_written);
        if (err)
            return err;
        if (strcmp(path, "-") != 0)
            fclose(f);
    } else {
#ifdef LIBMTX_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            f = gzdopen(STDOUT_FILENO, "w");
        } else if ((f = gzopen(path, "w")) == NULL) {
            return MTX_ERR_ERRNO;
        }
        err = mtxfile_gzwrite(mtxfile, f, format, bytes_written);
        if (err)
            return err;
        if (strcmp(path, "-") != 0)
            gzclose(f);
#else
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_fwrite()' writes a Matrix Market file to a stream.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_fwrite(
    const struct mtxfile * mtxfile,
    FILE * f,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    err = mtxfile_header_fwrite(&mtxfile->header, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_comments_fputs(&mtxfile->comments, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_size_fwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err)
        return err;
    err = mtxfile_data_fwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines,
        f, format, bytes_written);
    if (err)
        return err;
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_gzwrite()' writes a Matrix Market file to a
 * gzip-compressed stream.
 *
 * If `format' is `NULL', then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of `printf'. If the field
 * is `real', `double' or `complex', then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * `integer', then the format specifier must be '%d'. The format
 * string is ignored if the field is `pattern'. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtxfile_gzwrite(
    const struct mtxfile * mtxfile,
    gzFile f,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    err = mtxfile_header_gzwrite(&mtxfile->header, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_comments_gzputs(&mtxfile->comments, f, bytes_written);
    if (err)
        return err;
    err = mtxfile_size_gzwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err)
        return err;
    err = mtxfile_data_gzwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines,
        f, format, bytes_written);
    if (err)
        return err;
    return MTX_SUCCESS;
}
#endif

/*
 * Transpose and conjugate transpose.
 */

/**
 * `mtxfile_transpose()' tranposes a Matrix Market file.
 */
int mtxfile_transpose(
    struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object == mtxfile_matrix) {
        int64_t num_data_lines;
        err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
        if (err)
            return err;
        err = mtxfile_data_transpose(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, num_data_lines);
        if (err)
            return err;
        err = mtxfile_size_transpose(&mtxfile->size);
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
 * `mtxfile_conjugate_transpose()' tranposes and complex conjugates a
 * Matrix Market file.
 */
int mtxfile_conjugate_transpose(
    struct mtxfile * mtxfile);

/*
 * Sorting
 */

/**
 * `mtxfile_sort()' sorts a Matrix Market file in a given order.
 */
int mtxfile_sort(
    struct mtxfile * mtxfile,
    enum mtxfile_sorting sorting)
{
    int64_t num_data_lines;
    int err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err)
        return err;

    if (sorting == mtxfile_row_major) {
        return mtxfile_data_sort_row_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, num_data_lines);
    } else if (sorting == mtxfile_column_major) {
        return mtxfile_data_sort_column_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, num_data_lines);
    } else {
        return MTX_ERR_INVALID_MTX_SORTING;
    }
}

/*
 * Partitioning
 */

/**
 * `mtxfile_partition_rows()' partitions and reorders data lines of a
 * Matrix Market file according to the given row partitioning.
 *
 * The array `data_lines_per_part_ptr' must contain at least enough
 * storage for `row_partition->num_parts+1' values of type `int64_t'.
 * If successful, the `p'-th value of `data_lines_per_part_ptr' is an
 * offset to the first data line belonging to the `p'-th part of the
 * partition, while the final value of the array points to one place
 * beyond the final data line.
 */
int mtxfile_partition_rows(
    struct mtxfile * mtxfile,
    const struct mtx_partition * row_partition,
    int64_t * data_lines_per_part_ptr)
{
    int err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err)
        return err;

    int * row_parts = malloc(num_data_lines * sizeof(int));
    if (!row_parts)
        return MTX_ERR_ERRNO;

    /* Partition the data lines. */
    err = mtxfile_data_partition_rows(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        mtxfile->size.num_rows, mtxfile->size.num_columns,
        num_data_lines, 0,
        row_partition,
        row_parts);
    if (err) {
        free(row_parts);
        return err;
    }

    /* Sort the data lines according to the partitioning. */
    err = mtxfile_data_sort_by_key(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines, 0,
        row_parts);
    if (err) {
        free(row_parts);
        return err;
    }

    /* Calculate offset to the first element of each part. */
    for (int p = 0; p <= row_partition->num_parts; p++)
        data_lines_per_part_ptr[p] = 0;
    for (int64_t l = 0; l < num_data_lines; l++) {
        int part = row_parts[l];
        data_lines_per_part_ptr[part+1]++;
    }
    for (int p = 0; p < row_partition->num_parts; p++) {
        data_lines_per_part_ptr[p+1] +=
            data_lines_per_part_ptr[p];
    }

    free(row_parts);
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_from_row_partition()' creates a Matrix Market file
 * from a subset of the rows of another Matrix Market file.
 *
 * The array `data_lines_per_part_ptr' should have been obtained
 * previously by calling `mtxfile_partition_rows'.
 */
int mtxfile_init_from_row_partition(
    struct mtxfile * dst,
    const struct mtxfile * src,
    const struct mtx_partition * row_partition,
    int64_t * data_lines_per_part_ptr,
    int part)
{
    int err;
    err = mtxfile_header_copy(&dst->header, &src->header);
    if (err)
        return err;
    err = mtxfile_comments_copy(&dst->comments, &src->comments);
    if (err)
        return err;
    err = mtxfile_size_copy(&dst->size, &src->size);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    dst->precision = src->precision;

    if (dst->header.format == mtxfile_array) {
        dst->size.num_rows = row_partition->size_per_part[part];
    } else if (dst->header.format == mtxfile_coordinate) {
        dst->size.num_nonzeros =
            data_lines_per_part_ptr[part+1] - data_lines_per_part_ptr[part];
    } else {
        mtxfile_comments_free(&dst->comments);
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&dst->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    if (num_data_lines != data_lines_per_part_ptr[part+1]-data_lines_per_part_ptr[part])
    {
        mtxfile_comments_free(&dst->comments);
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    err = mtxfile_data_alloc(
        &dst->data,
        dst->header.object,
        dst->header.format,
        dst->header.field,
        dst->precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    err = mtxfile_data_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        num_data_lines, 0, data_lines_per_part_ptr[part]);
    if (err) {
        mtxfile_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxfile_send()' sends a Matrix Market file to another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxfile_recv()'.
 */
int mtxfile_send(
    const struct mtxfile * mtxfile,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxfile_header_send(&mtxfile->header, dest, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfile_comments_send(&mtxfile->comments, dest, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfile_size_send(&mtxfile->size, dest, tag, comm, mpierror);
    if (err)
        return err;
    mpierror->mpierrcode = MPI_Send(
        &mtxfile->precision, 1, MPI_INT, dest, tag, comm);
    if (mpierror->mpierrcode)
        return MTX_ERR_MPI;

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err)
        return err;

    err = mtxfile_data_send(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        num_data_lines, 0,
        dest, tag, comm, mpierror);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_recv()' receives a Matrix Market file from another MPI
 * process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxfile_send()'.
 */
int mtxfile_recv(
    struct mtxfile * mtxfile,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxfile_header_recv(&mtxfile->header, source, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfile_comments_recv(&mtxfile->comments, source, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfile_size_recv(&mtxfile->size, source, tag, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }
    mpierror->mpierrcode = MPI_Recv(
        &mtxfile->precision, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->mpierrcode) {
        mtxfile_comments_free(&mtxfile->comments);
        return MTX_ERR_MPI;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }

    err = mtxfile_data_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }

    err = mtxfile_data_recv(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        num_data_lines, 0,
        source, tag, comm, mpierror);
    if (err) {
        mtxfile_data_free(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision);
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_bcast()' broadcasts a Matrix Market file from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to `mtxfile_bcast()'.
 */
int mtxfile_bcast(
    struct mtxfile * mtxfile,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    err = mtxfile_header_bcast(&mtxfile->header, root, comm, mpierror);
    if (err)
        return err;
    err = mtxfile_comments_bcast(&mtxfile->comments, root, comm, mpierror);
    if (err)
        return err;
    err = mtxfile_size_bcast(&mtxfile->size, root, comm, mpierror);
    if (err) {
        if (rank != root)
            mtxfile_comments_free(&mtxfile->comments);
        return err;
    }
    mpierror->mpierrcode = MPI_Bcast(
        &mtxfile->precision, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank != root)
            mtxfile_comments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &mtxfile->size, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank != root)
            mtxfile_comments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    if (rank != root) {
        err = mtxfile_data_alloc(
            &mtxfile->data,
            mtxfile->header.object,
            mtxfile->header.format,
            mtxfile->header.field,
            mtxfile->precision, num_data_lines);
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank != root)
            mtxfile_comments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfile_data_bcast(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision,
        num_data_lines, 0,
        root, comm, mpierror);
    if (err) {
        if (rank != root) {
            mtxfile_data_free(
                &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
                mtxfile->header.field, mtxfile->precision);
            mtxfile_comments_free(&mtxfile->comments);
        }
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_scatterv()' scatters a Matrix Market file from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to `MPI_Scatterv()' and requires every process in
 * the communicator to perform matching calls to `mtxfile_scatterv()'.
 */
int mtxfile_scatterv(
    const struct mtxfile * sendmtxfile,
    int * sendcounts,
    int * displs,
    struct mtxfile * recvmtxfile,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    err = (rank == root)
        ? mtxfile_header_copy(&recvmtxfile->header, &sendmtxfile->header)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_header_bcast(&recvmtxfile->header, root, comm, mpierror);
    if (err)
        return err;

    err = (rank == root)
        ? mtxfile_comments_init(&recvmtxfile->comments)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == root)
        ? mtxfile_comments_copy(&recvmtxfile->comments, &sendmtxfile->comments)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfile_comments_bcast(&recvmtxfile->comments, root, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&recvmtxfile->comments);
        return err;
    }

    err = mtxfile_size_scatterv(
        &sendmtxfile->size, &recvmtxfile->size,
        recvmtxfile->header.object, recvmtxfile->header.format,
        recvcount, root, comm, mpierror);
    if (err) {
        mtxfile_comments_free(&recvmtxfile->comments);
        return err;
    }

    recvmtxfile->precision = sendmtxfile->precision;
    mpierror->mpierrcode = MPI_Bcast(
        &recvmtxfile->precision, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&recvmtxfile->size, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = recvcount != num_data_lines ? MTX_ERR_INDEX_OUT_OF_BOUNDS : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfile_data_alloc(
        &recvmtxfile->data,
        recvmtxfile->header.object,
        recvmtxfile->header.format,
        recvmtxfile->header.field,
        recvmtxfile->precision,
        recvcount);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_comments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfile_data_scatterv(
        &sendmtxfile->data,
        recvmtxfile->header.object, recvmtxfile->header.format,
        recvmtxfile->header.field, recvmtxfile->precision,
        0, sendcounts, displs,
        &recvmtxfile->data, 0, recvcount,
        root, comm, mpierror);
    if (err) {
        mtxfile_data_free(
            &recvmtxfile->data, recvmtxfile->header.object, recvmtxfile->header.format,
            recvmtxfile->header.field, recvmtxfile->precision);
        mtxfile_comments_free(&recvmtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_distribute_rows()' partitions and distributes rows of a
 * Matrix Market file from an MPI root process to other processes in a
 * communicator.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to `mtxfile_distribute_rows()'.
 *
 * `row_partition' must be a partitioning of the rows of the matrix or
 * vector represented by `src'.
 */
int mtxfile_distribute_rows(
    struct mtxfile * dst,
    struct mtxfile * src,
    const struct mtx_partition * row_partition,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Partition the rows. */
    int64_t * data_lines_per_part_ptr =
        (rank == root) ? malloc((comm_size+1) * sizeof(int64_t)) : NULL;
    err = (rank == root && !data_lines_per_part_ptr) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == root)
        ? mtxfile_partition_rows(src, row_partition, data_lines_per_part_ptr)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(data_lines_per_part_ptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Find the number of data lines and offsets for each part. */
    int * sendcounts = (rank == root) ? malloc(2*comm_size * sizeof(int)) : NULL;
    err = (rank == root && !sendcounts) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(data_lines_per_part_ptr);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * displs = (rank == root) ? &sendcounts[comm_size] : NULL;
    if (rank == root) {
        for (int p = 0; p < comm_size; p++) {
            sendcounts[p] = data_lines_per_part_ptr[p+1] - data_lines_per_part_ptr[p];
            displs[p] = data_lines_per_part_ptr[p];
        }
        free(data_lines_per_part_ptr);
    }

    int recvcount;
    mpierror->mpierrcode = MPI_Scatter(
        sendcounts, 1, MPI_INT, &recvcount, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            free(sendcounts);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Scatter the Matrix Market file. */
    err = mtxfile_scatterv(
        src, sendcounts, displs, dst, recvcount, root, comm, mpierror);
    if (err) {
        if (rank == root)
            free(sendcounts);
        return err;
    }

    if (rank == root)
        free(sendcounts);
    return MTX_SUCCESS;
}

/**
 * `mtxfile_buffer_size()' configures a size line for a Matrix Market
 * file that can be used as a temporary buffer for reading on an MPI
 * root process and distributing to all the processes in a
 * communicator.  The buffer will be no greater than the given buffer
 * size `bufsize' in bytes.
 */
static int mtxfile_buffer_size(
    struct mtxfile_size * size,
    enum mtxfile_object object,
    enum mtxfile_format format,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    size_t bufsize,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    size_t size_per_data_line;
    err = mtxfile_data_size_per_element(
        &size_per_data_line, object, format,
        field, precision);
    if (err)
        return err;
    int64_t num_data_lines = bufsize / size_per_data_line;

    if (format == mtxfile_array) {
        int comm_size;
        *mpierrcode = MPI_Comm_size(comm, &comm_size);
        if (*mpierrcode)
            return MTX_ERR_MPI;

        if (object == mtxfile_matrix) {
            /* Round to a multiple of the product of the number of
             * columns and the number of processes, since we need
             * to read the same number of entire rows per process
             * to correctly distribute the data. */
            size->num_rows =
                (num_data_lines / (num_columns*comm_size)) * (num_columns*comm_size);
            if (size->num_rows > num_rows)
                size->num_rows = num_rows;
            size->num_columns = num_columns;
            size->num_nonzeros = -1;
        } else if (object == mtxfile_vector) {
            /* Round to a multiple of the number of processes,
             * since we need to read the same number of lines per
             * process to correctly distribute the data. */
            size->num_rows =
                ((num_data_lines / comm_size) * comm_size < num_rows) ?
                (num_data_lines / comm_size) * comm_size : num_rows;
            size->num_columns = -1;
            size->num_nonzeros = -1;
        } else {
            return MTX_ERR_INVALID_MTX_OBJECT;
        }
        if (size->num_rows <= 0)
            return MTX_ERR_NO_BUFFER_SPACE;
    } else if (format == mtxfile_coordinate) {
        size->num_rows = num_rows;
        size->num_columns = num_columns;
        size->num_nonzeros =
            (num_data_lines < num_nonzeros) ? num_data_lines : num_nonzeros;
        if (size->num_nonzeros <= 0)
            return MTX_ERR_NO_BUFFER_SPACE;
    } else {
        return MTX_ERR_INVALID_MTX_FORMAT;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_fread_distribute_rows()' reads a Matrix Market file from a
 * stream and distributes the rows of the underlying matrix or vector
 * among MPI processes in a communicator.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 *
 * For a matrix or vector in array format, `bufsize' must be at least
 * large enough to fit one row per MPI process in the communicator.
 */
int mtxfile_fread_distribute_rows(
    struct mtxfile * mtxfile,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtx_precision precision,
    enum mtx_partition_type row_partition_type,
    size_t bufsize,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;

    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    bool free_linebuf = (rank == root) && !linebuf;
    if (rank == root && !linebuf) {
        line_max = sysconf(_SC_LINE_MAX);
        linebuf = malloc(line_max+1);
    }
    err = (rank == root && !linebuf) ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    /* Read the header on the root process */
    struct mtxfile rootmtx;
    if (rank == root) {
        err = mtxfile_fread_header(
            &rootmtx.header, f, lines_read, bytes_read, line_max, linebuf);
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read comments on the root process */
    err = (rank == root) ? mtxfile_fread_comments(
        &rootmtx.comments, f, lines_read, bytes_read, line_max, linebuf)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Read the size line on the root process into a temporary
     * location. */
    struct mtxfile_size size;
    err = (rank == root) ? mtxfile_fread_size(
        &size, f, lines_read, bytes_read, line_max, linebuf,
        rootmtx.header.object, rootmtx.header.format)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            mtxfile_comments_free(&rootmtx.comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    rootmtx.precision = precision;

    /* Partition the rows of the matrix or vector. */
    struct mtx_partition row_partition;
    err = (rank == root) ? mtx_partition_init(
        &row_partition, row_partition_type, size.num_rows, comm_size, 0)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank == root)
            mtxfile_comments_free(&rootmtx.comments);
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Distribute the initial, empty matrix. */
    if (rank == root) {
        rootmtx.size.num_rows = (size.num_nonzeros >= 0) ? size.num_rows : 0;
        rootmtx.size.num_columns = size.num_columns;
        rootmtx.size.num_nonzeros = (size.num_nonzeros >= 0) ? 0 : -1;
    }
    err = mtxfile_distribute_rows(
        mtxfile, &rootmtx, &row_partition, root, comm, mpierror);
    if (err) {
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_comments_free(&rootmtx.comments);
        }
        if (free_linebuf)
            free(linebuf);
        return err;
    }

    /* Configure the size line for the temporary Matrix Market file to
     * be used as a buffer for reading on the root process. */
    err = (rank == root)
        ? mtxfile_buffer_size(
            &rootmtx.size, rootmtx.header.object, rootmtx.header.format,
            rootmtx.header.field, precision,
            size.num_rows, size.num_columns, size.num_nonzeros,
            bufsize, comm, &mpierror->mpierrcode)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(mtxfile);
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_comments_free(&rootmtx.comments);
        }
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t num_data_lines_rootmtx;
    err = (rank == root)
        ? mtxfile_size_num_data_lines(&rootmtx.size, &num_data_lines_rootmtx)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(mtxfile);
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_comments_free(&rootmtx.comments);
        }
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mpierror->mpierrcode = MPI_Bcast(
        &num_data_lines_rootmtx, 1, MPI_INT64_T, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(mtxfile);
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_comments_free(&rootmtx.comments);
        }
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Allocate a temporary Matrix Market file on the root. */
    err = (rank == root) ?
        mtxfile_data_alloc(
            &rootmtx.data, rootmtx.header.object, rootmtx.header.format,
            rootmtx.header.field, rootmtx.precision, num_data_lines_rootmtx)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(mtxfile);
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_comments_free(&rootmtx.comments);
        }
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* Determine the total number of data lines. */
    int64_t num_data_lines;
    err = (rank == root)
        ? mtxfile_size_num_data_lines(&size, &num_data_lines)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(mtxfile);
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_free(&rootmtx);
        }
        if (free_linebuf)
            free(linebuf);
        return err;
    }
    mpierror->mpierrcode = MPI_Bcast(
        &num_data_lines, 1, MPI_INT64_T, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfile_free(mtxfile);
        if (rank == root) {
            mtx_partition_free(&row_partition);
            mtxfile_free(&rootmtx);
        }
        if (free_linebuf)
            free(linebuf);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines_remaining = num_data_lines;
    while (num_data_lines_remaining > 0) {
        int64_t num_data_lines_to_read =
            (num_data_lines_rootmtx < num_data_lines_remaining)
            ? num_data_lines_rootmtx : num_data_lines_remaining;
        if (rank == root && rootmtx.size.num_nonzeros >= 0)
            rootmtx.size.num_nonzeros = num_data_lines_to_read;

        /* Read the next set of data lines on the root process. */
        if (rank == root) {
            err = mtxfile_fread_data(
                &rootmtx.data, f, lines_read, bytes_read, line_max, linebuf,
                rootmtx.header.object,
                rootmtx.header.format,
                rootmtx.header.field,
                rootmtx.precision,
                rootmtx.size.num_rows,
                rootmtx.size.num_columns,
                num_data_lines_to_read);
        }
        if (mtxmpierror_allreduce(mpierror, err)) {
            mtxfile_free(mtxfile);
            if (rank == root) {
                mtx_partition_free(&row_partition);
                mtxfile_free(&rootmtx);
            }
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Distribute the matrix rows. */
        struct mtxfile tmpmtx;
        err = mtxfile_distribute_rows(
            &tmpmtx, &rootmtx, &row_partition, root, comm, mpierror);
        if (err) {
            mtxfile_free(mtxfile);
            if (rank == root) {
                mtx_partition_free(&row_partition);
                mtxfile_free(&rootmtx);
            }
            if (free_linebuf)
                free(linebuf);
            return err;
        }

        /* Concatenate the newly distributed matrices with the
         * existing ones. */
        err = mtxfile_cat(mtxfile, &tmpmtx);
        if (mtxmpierror_allreduce(mpierror, err)) {
            mtxfile_free(&tmpmtx);
            mtxfile_free(mtxfile);
            if (rank == root) {
                mtx_partition_free(&row_partition);
                mtxfile_free(&rootmtx);
            }
            if (free_linebuf)
                free(linebuf);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        mtxfile_free(&tmpmtx);
        num_data_lines_remaining -= num_data_lines_to_read;
    }

    if (rank == root) {
        mtx_partition_free(&row_partition);
        mtxfile_free(&rootmtx);
    }
    if (free_linebuf)
        free(linebuf);
    return MTX_SUCCESS;
}
#endif
