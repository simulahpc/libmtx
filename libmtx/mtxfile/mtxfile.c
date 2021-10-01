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
 * `mtxfile_alloc()' allocates storage for a Matrix Market file with
 * the given header line, comment lines and size line.
 *
 * ‘comments’ may be ‘NULL’, in which case it is ignored.
 */
int mtxfile_alloc(
    struct mtxfile * mtxfile,
    const struct mtxfile_header * header,
    const struct mtxfile_comments * comments,
    const struct mtxfile_size * size,
    enum mtx_precision precision)
{
    int err;
    err = mtxfile_header_copy(&mtxfile->header, header);
    if (err)
        return err;
    if (comments) {
        err = mtxfile_comments_copy(&mtxfile->comments, comments);
        if (err)
            return err;
    } else {
        err = mtxfile_comments_init(&mtxfile->comments);
        if (err)
            return err;
    }
    err = mtxfile_size_copy(&mtxfile->size, size);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }
    mtxfile->precision = precision;

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }

    err = mtxfile_data_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&mtxfile->comments);
        return err;
    }
    return MTX_SUCCESS;
}

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
 * `mtxfile_alloc_copy()' allocates storage for a copy of a Matrix
 * Market file without initialising the underlying values.
 */
int mtxfile_alloc_copy(
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
    dst->precision = src->precision;

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(
        &src->size, &num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }

    err = mtxfile_data_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_data_lines);
    if (err) {
        mtxfile_comments_free(&dst->comments);
        return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_copy()' creates a copy of a Matrix Market file.
 */
int mtxfile_init_copy(
    struct mtxfile * dst,
    const struct mtxfile * src)
{
    int err;
    err = mtxfile_alloc_copy(dst, src);
    if (err)
        return err;
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
 *
 * If ‘skip_comments’ is ‘true’, then comment lines from ‘src’ are not
 * concatenated to those of ‘dst’.
 */
int mtxfile_cat(
    struct mtxfile * dst,
    const struct mtxfile * src,
    bool skip_comments)
{
    int err;
    if (dst->header.object != src->header.object)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (dst->header.format != src->header.format)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
    if (dst->header.field != src->header.field)
        return MTX_ERR_INCOMPATIBLE_MTX_FIELD;
    if (dst->header.symmetry != src->header.symmetry)
        return MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY;

    if (!skip_comments) {
        err = mtxfile_comments_cat(&dst->comments, &src->comments);
        if (err)
            return err;
    }

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

/**
 * `mtxfile_catn()' concatenates multiple Matrix Market files.
 *
 * The files must have identical header lines. Furthermore, for
 * matrices in array format, all matrices must have the same number of
 * columns, since entire rows are concatenated.  For matrices or
 * vectors in coordinate format, the number of rows and columns must
 * be the same.
 *
 * If ‘skip_comments’ is ‘true’, then comment lines from ‘src’ are not
 * concatenated to those of ‘dst’.
 */
int mtxfile_catn(
    struct mtxfile * dst,
    int num_srcs,
    const struct mtxfile * srcs,
    bool skip_comments)
{
    int err;
    for (int i = 0; i < num_srcs; i++) {
        const struct mtxfile * src = &srcs[i];
        if (dst->header.object != src->header.object)
            return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
        if (dst->header.format != src->header.format)
            return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
        if (dst->header.field != src->header.field)
            return MTX_ERR_INCOMPATIBLE_MTX_FIELD;
        if (dst->header.symmetry != src->header.symmetry)
            return MTX_ERR_INCOMPATIBLE_MTX_SYMMETRY;
    }

    if (!skip_comments) {
        for (int i = 0; i < num_srcs; i++) {
            const struct mtxfile * src = &srcs[i];
            err = mtxfile_comments_cat(&dst->comments, &src->comments);
            if (err)
                return err;
        }
    }

    int64_t num_data_lines_dst;
    err = mtxfile_size_num_data_lines(&dst->size, &num_data_lines_dst);
    if (err)
        return err;

    for (int i = 0; i < num_srcs; i++) {
        const struct mtxfile * src = &srcs[i];
        int64_t num_data_lines_src;
        err = mtxfile_size_num_data_lines(&src->size, &num_data_lines_src);
        if (err)
            return err;

        err = mtxfile_size_cat(
            &dst->size, &src->size, dst->header.object, dst->header.format);
        if (err)
            return err;
    }

    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&dst->size, &num_data_lines);
    if (err)
        return err;

    union mtxfile_data data;
    err = mtxfile_data_alloc(
        &data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_data_lines);
    if (err)
        return err;

    int64_t dstoffset = 0;
    int64_t srcoffset = 0;
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
    dstoffset += num_data_lines_dst;

    for (int i = 0; i < num_srcs; i++) {
        const struct mtxfile * src = &srcs[i];
        int64_t num_data_lines_src;
        err = mtxfile_size_num_data_lines(&src->size, &num_data_lines_src);
        if (err)
            return err;

        err = mtxfile_data_copy(
            &data, &src->data,
            dst->header.object, dst->header.format,
            dst->header.field, dst->precision,
            num_data_lines_src, dstoffset, 0);
        if (err) {
            mtxfile_data_free(
                &data, dst->header.object, dst->header.format,
                dst->header.field, dst->precision);
            return err;
        }
        dstoffset += num_data_lines_src;
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
    const float * data)
{
    int err;
    err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_real, symmetry, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_free(mtxfile);
        return err;
    }
    memcpy(mtxfile->data.array_real_single, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

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
    const float (* data)[2])
{
    int err;
    err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_complex, symmetry, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_free(mtxfile);
        return err;
    }
    memcpy(mtxfile->data.array_complex_single, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

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
    const double (* data)[2])
{
    int err;
    err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_complex, symmetry, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_free(mtxfile);
        return err;
    }
    memcpy(mtxfile->data.array_complex_double, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

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
    const int32_t * data)
{
    int err;
    err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_integer, symmetry, mtx_single, num_rows, num_columns);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_free(mtxfile);
        return err;
    }
    memcpy(mtxfile->data.array_integer_single, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

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
    const int64_t * data)
{
    int err;
    err = mtxfile_alloc_matrix_array(
        mtxfile, mtxfile_integer, symmetry, mtx_double, num_rows, num_columns);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err) {
        mtxfile_free(mtxfile);
        return err;
    }
    memcpy(mtxfile->data.array_integer_double, data, num_data_lines * sizeof(*data));
    return MTX_SUCCESS;
}

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
    const int32_t * data)
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_integer, mtx_single, num_rows);
    if (err)
        return err;
    memcpy(mtxfile->data.array_integer_single, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

/**
 * `mtxfile_init_vector_array_integer_double()' allocates and
 * initialises a vector in array format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_array_integer_double(
    struct mtxfile * mtxfile,
    int num_rows,
    const int64_t * data)
{
    int err = mtxfile_alloc_vector_array(
        mtxfile, mtxfile_integer, mtx_double, num_rows);
    if (err)
        return err;
    memcpy(mtxfile->data.array_integer_double, data, num_rows * sizeof(*data));
    return MTX_SUCCESS;
}

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
 * Modifying values
 */

/**
 * `mtxfile_set_constant_real_single()' sets every (nonzero) value of
 * a matrix or vector equal to a constant, single precision floating
 * point number.
 */
int mtxfile_set_constant_real_single(
    struct mtxfile * mtxfile,
    float a)
{
    int err;
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile->size, &num_data_lines);
    if (err)
        return err;
    return mtxfile_data_set_constant_real_single(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        num_data_lines, 0, a);
}

/**
 * `mtxfile_set_constant_real_double()' sets every (nonzero) value of
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
 * `mtxfile_set_constant_complex_single()' sets every (nonzero) value
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
 * `mtxfile_set_constant_integer_single()' sets every (nonzero) value
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
        num_data_lines, 0);
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
        num_data_lines, 0);
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
 * The file may optionally be compressed by gzip.
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
        err = mtxfile_fwrite(mtxfile, f, format, bytes_written);
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
        err = mtxfile_gzwrite(mtxfile, f, format, bytes_written);
        if (err) {
            gzclose(f);
            return err;
        }
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
 * ‘mtxfile_init_from_partition()’ creates Matrix Market files for
 * each part of a partitioning of another Matrix Market file.
 *
 * ‘dst’ must point to an array of type ‘struct mtxfile’ whose length
 * is equal to the number of parts in the partitioning, num_parts’.
 * The ‘p’th entry in the array will be a Matrix Market file
 * containing the ‘p’th part of the original Matrix Market file,
 * ‘src’, according to the partitioning given by ‘row_partition’.
 *
 * The ‘p’th value of ‘data_lines_per_part_ptr’ must be an offset to
 * the first data line belonging to the ‘p’th part of the partition,
 * while the final value of the array points to one place beyond the
 * final data line.  Moreover for each part ‘p’ of the partitioning,
 * the entries from ‘data_lines_per_part[p]’ up to, but not including,
 * ‘data_lines_per_part[p+1]’, are the indices of the data lines in
 * ‘src’ that are assigned to the ‘p’th part of the partitioning.
 */
int mtxfile_init_from_partition(
    struct mtxfile * dst,
    const struct mtxfile * src,
    int num_parts,
    const int64_t * data_lines_per_part_ptr,
    const int64_t * data_lines_per_part)
{
    int err;
    for (int p = 0; p < num_parts; p++) {
        struct mtxfile_size size;
        size.num_rows = src->size.num_rows;
        size.num_columns = src->size.num_columns;
        size.num_nonzeros = src->size.num_nonzeros;
        int64_t N = data_lines_per_part_ptr[p+1] - data_lines_per_part_ptr[p];
        if (src->size.num_nonzeros >= 0) {
            size.num_nonzeros = N;
        } else if (src->size.num_columns > 0) {
            size.num_rows = (N + src->size.num_columns-1) / src->size.num_columns;
        } else if (src->size.num_rows >= 0) {
            size.num_rows = N;
        }

        err = mtxfile_alloc(
            &dst[p], &src->header, &src->comments, &size, src->precision);
        if (err) {
            mtxfile_comments_free(&dst[p].comments);
            return err;
        }

        const int64_t * srcdispls = &data_lines_per_part[data_lines_per_part_ptr[p]];
        err = mtxfile_data_copy_gather(
            &dst[p].data, &src->data,
            dst[p].header.object, dst[p].header.format,
            dst[p].header.field, dst[p].precision,
            N, 0, srcdispls);
        if (err) {
            for (int q = p; q >= 0; q--)
                mtxfile_free(&dst[q]);
            return err;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxfile_partition_rows()’ partitions data lines of a Matrix Market
 * file according to the given row partitioning.
 *
 * If it is not ‘NULL’, the array ‘part_per_data_line’ must contain
 * enough storage to hold one ‘int’ for each data line. (The number of
 * data lines is obtained from ‘mtxfile_size_num_data_lines()’). On a
 * successful return, the ‘k’th entry in the array specifies the part
 * number that was assigned to the ‘k’th data line of ‘src’.
 *
 * The array ‘data_lines_per_part_ptr’ must contain at least enough
 * storage for ‘row_partition->num_parts+1’ values of type ‘int64_t’.
 * If successful, the ‘p’th value of ‘data_lines_per_part_ptr’ is an
 * offset to the first data line belonging to the ‘p’th part of the
 * partition, while the final value of the array points to one place
 * beyond the final data line.  Moreover ‘data_lines_per_part’ must
 * contain enough storage to hold one ‘int64_t’ for each data line.
 * For each part ‘p’ of the partitioning, the entries from
 * ‘data_lines_per_part[p]’ up to, but not including,
 * ‘data_lines_per_part[p+1]’, are the indices of the data lines in
 * ‘src’ that are assigned to the ‘p’th part of the partitioning.
 */
int mtxfile_partition_rows(
    const struct mtxfile * mtxfile,
    int64_t size,
    int64_t offset,
    const struct mtx_partition * row_partition,
    int * part_per_data_line,
    int64_t * data_lines_per_part_ptr,
    int64_t * data_lines_per_part)
{
    int err;
    bool alloc_part_per_data_line = !part_per_data_line;
    if (alloc_part_per_data_line) {
        part_per_data_line = malloc(size * sizeof(int));
        if (!part_per_data_line)
            return MTX_ERR_ERRNO;
    }
    err = mtxfile_data_partition_rows(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision,
        mtxfile->size.num_rows, mtxfile->size.num_columns,
        size, offset, row_partition,
        part_per_data_line);
    if (err) {
        if (alloc_part_per_data_line)
            free(part_per_data_line);
        return err;
    }

    for (int p = 0; p <= row_partition->num_parts; p++)
        data_lines_per_part_ptr[p] = 0;
    for (int64_t k = 0; k < size; k++) {
        int part = part_per_data_line[k];
        data_lines_per_part_ptr[part+1]++;
    }
    for (int p = 0; p < row_partition->num_parts; p++) {
        data_lines_per_part_ptr[p+1] +=
            data_lines_per_part_ptr[p];
    }
    for (int64_t k = 0; k < size; k++) {
        int part = part_per_data_line[k];
        int64_t l = data_lines_per_part_ptr[part];
        data_lines_per_part[l] = k;
        data_lines_per_part_ptr[part]++;
    }
    for (int p = row_partition->num_parts-1; p >= 0; p--) {
        data_lines_per_part_ptr[p+1] =
            data_lines_per_part_ptr[p];
    }
    data_lines_per_part_ptr[0] = 0;

    if (alloc_part_per_data_line)
        free(part_per_data_line);
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
        dst->size.num_rows = row_partition->index_sets[part].size;
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
 * `mtxfile_gather()' gathers Matrix Market files onto an MPI root
 * process from other processes in a communicator.
 *
 * This is analogous to `MPI_Gather()' and requires every process in
 * the communicator to perform matching calls to `mtxfile_gather()'.
 */
int mtxfile_gather(
    const struct mtxfile * sendmtxfile,
    struct mtxfile * recvmtxfiles,
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

    for (int p = 0; p < comm_size; p++) {
        /* Send to the root process */
        err = (rank != root && rank == p)
            ? mtxfile_send(sendmtxfile, root, 0, comm, mpierror)
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root) {
                for (int q = p-1; q >= 0; q--)
                    mtxfile_free(&recvmtxfiles[q]);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Receive on the root process */
        err = (rank == root && p != root)
            ? mtxfile_recv(&recvmtxfiles[p], p, 0, comm, mpierror)
            : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank == root) {
                for (int q = p-1; q >= 0; q--)
                    mtxfile_free(&recvmtxfiles[q]);
            }
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* Perform a copy on the root process */
        err = (rank == root && p == root)
            ? mtxfile_init_copy(&recvmtxfiles[p], sendmtxfile) : MTX_SUCCESS;
        if (mtxmpierror_allreduce(mpierror, err)) {
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
 * `mtxfile_allgather()' gathers Matrix Market files onto every MPI
 * process from other processes in a communicator.
 *
 * This is analogous to `MPI_Allgather()' and requires every process
 * in the communicator to perform matching calls to
 * `mtxfile_allgather()'.
 */
int mtxfile_allgather(
    const struct mtxfile * sendmtxfile,
    struct mtxfile * recvmtxfiles,
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
    for (int p = 0; p < comm_size; p++) {
        err = mtxfile_gather(sendmtxfile, recvmtxfiles, p, comm, mpierror);
        if (err)
            return err;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_scatter()' scatters Matrix Market files from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to `MPI_Scatter()' and requires every process in
 * the communicator to perform matching calls to `mtxfile_scatter()'.
 */
int mtxfile_scatter(
    const struct mtxfile * sendmtxfiles,
    struct mtxfile * recvmtxfile,
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

    for (int p = 0; p < comm_size; p++) {
        if (rank == root && p != root) {
            /* Send from the root process */
            err = mtxfile_send(&sendmtxfiles[p], p, 0, comm, mpierror);
        } else if (rank != root && rank == p) {
            /* Receive from the root process */
            err = mtxfile_recv(recvmtxfile, root, 0, comm, mpierror);
        } else if (rank == root && p == root) {
            /* Perform a copy on the root process */
            err = mtxfile_init_copy(recvmtxfile, &sendmtxfiles[p]);
        } else {
            err = MTX_SUCCESS;
        }
        if (mtxmpierror_allreduce(mpierror, err)) {
            if (rank < p)
                mtxfile_free(recvmtxfile);
            return MTX_ERR_MPI_COLLECTIVE;
        }
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_alltoall()' performs an all-to-all exchange of Matrix
 * Market files between MPI process in a communicator.
 *
 * This is analogous to `MPI_Alltoall()' and requires every process in
 * the communicator to perform matching calls to `mtxfile_alltoall()'.
 */
int mtxfile_alltoall(
    const struct mtxfile * sendmtxfiles,
    struct mtxfile * recvmtxfiles,
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
    for (int p = 0; p < comm_size; p++) {
        err = mtxfile_scatter(sendmtxfiles, &recvmtxfiles[p], p, comm, mpierror);
        if (err) {
            for (int q = p-1; q >= 0; q--)
                mtxfile_free(&recvmtxfiles[q]);
            return err;
        }
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
    const int * sendcounts,
    const int * displs,
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
#endif
