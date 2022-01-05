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
 * `mtxfile_alloc()' allocates storage for a Matrix Market file with
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
    if (err)
        return err;
    if (comments) {
        err = mtxfilecomments_copy(&mtxfile->comments, comments);
        if (err)
            return err;
    } else {
        err = mtxfilecomments_init(&mtxfile->comments);
        if (err)
            return err;
    }
    err = mtxfilesize_copy(&mtxfile->size, size);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    mtxfile->precision = precision;

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        size, header->symmetry, &num_data_lines);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }

    err = mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
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
    mtxfiledata_free(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision);
    mtxfilecomments_free(&mtxfile->comments);
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
    err = mtxfileheader_copy(&dst->header, &src->header);
    if (err)
        return err;
    err = mtxfilecomments_copy(&dst->comments, &src->comments);
    if (err)
        return err;
    err = mtxfilesize_copy(&dst->size, &src->size);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    dst->precision = src->precision;

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &src->size, src->header.symmetry, &num_data_lines);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }

    err = mtxfiledata_alloc(
        &dst->data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_data_lines);
    if (err) {
        mtxfilecomments_free(&dst->comments);
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
    err = mtxfilesize_num_data_lines(
        &src->size, src->header.symmetry, &num_data_lines);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    err = mtxfiledata_copy(
        &dst->data, &src->data,
        src->header.object, src->header.format,
        src->header.field, src->precision,
        num_data_lines, 0, 0);
    if (err) {
        mtxfilecomments_free(&dst->comments);
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
        err = mtxfilecomments_cat(&dst->comments, &src->comments);
        if (err)
            return err;
    }

    int64_t num_data_lines_dst;
    err = mtxfilesize_num_data_lines(
        &dst->size, dst->header.symmetry, &num_data_lines_dst);
    if (err)
        return err;
    int64_t num_data_lines_src;
    err = mtxfilesize_num_data_lines(
        &src->size, src->header.symmetry, &num_data_lines_src);
    if (err)
        return err;

    err = mtxfilesize_cat(
        &dst->size, &src->size, dst->header.object, dst->header.format);
    if (err)
        return err;

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &dst->size, dst->header.symmetry, &num_data_lines);
    if (err)
        return err;
    if (num_data_lines_dst + num_data_lines_src != num_data_lines)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    union mtxfiledata data;
    err = mtxfiledata_alloc(
        &data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_data_lines);
    if (err)
        return err;

    err = mtxfiledata_copy(
        &data, &dst->data,
        dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        num_data_lines_dst, 0, 0);
    if (err) {
        mtxfiledata_free(
            &data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision);
        return err;
    }

    err = mtxfiledata_copy(
        &data, &src->data,
        dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        num_data_lines_src, num_data_lines_dst, 0);
    if (err) {
        mtxfiledata_free(
            &data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision);
        return err;
    }

    union mtxfiledata olddata = dst->data;
    dst->data = data;
    mtxfiledata_free(
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
            err = mtxfilecomments_cat(&dst->comments, &src->comments);
            if (err)
                return err;
        }
    }

    int64_t num_data_lines_dst;
    err = mtxfilesize_num_data_lines(
        &dst->size, dst->header.symmetry, &num_data_lines_dst);
    if (err)
        return err;

    for (int i = 0; i < num_srcs; i++) {
        const struct mtxfile * src = &srcs[i];
        int64_t num_data_lines_src;
        err = mtxfilesize_num_data_lines(
            &src->size, src->header.symmetry, &num_data_lines_src);
        if (err)
            return err;

        err = mtxfilesize_cat(
            &dst->size, &src->size, dst->header.object, dst->header.format);
        if (err)
            return err;
    }

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &dst->size, dst->header.symmetry, &num_data_lines);
    if (err)
        return err;

    union mtxfiledata data;
    err = mtxfiledata_alloc(
        &data, dst->header.object, dst->header.format,
        dst->header.field, dst->precision, num_data_lines);
    if (err)
        return err;

    int64_t dstoffset = 0;
    int64_t srcoffset = 0;
    err = mtxfiledata_copy(
        &data, &dst->data,
        dst->header.object, dst->header.format,
        dst->header.field, dst->precision,
        num_data_lines_dst, 0, 0);
    if (err) {
        mtxfiledata_free(
            &data, dst->header.object, dst->header.format,
            dst->header.field, dst->precision);
        return err;
    }
    dstoffset += num_data_lines_dst;

    for (int i = 0; i < num_srcs; i++) {
        const struct mtxfile * src = &srcs[i];
        int64_t num_data_lines_src;
        err = mtxfilesize_num_data_lines(
            &src->size, src->header.symmetry, &num_data_lines_src);
        if (err)
            return err;

        err = mtxfiledata_copy(
            &data, &src->data,
            dst->header.object, dst->header.format,
            dst->header.field, dst->precision,
            num_data_lines_src, dstoffset, 0);
        if (err) {
            mtxfiledata_free(
                &data, dst->header.object, dst->header.format,
                dst->header.field, dst->precision);
            return err;
        }
        dstoffset += num_data_lines_src;
    }

    union mtxfiledata olddata = dst->data;
    dst->data = data;
    mtxfiledata_free(
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
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
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
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = num_columns;
    mtxfile->size.num_nonzeros = -1;
    mtxfile->precision = precision;

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;

    err = mtxfiledata_alloc(
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
    enum mtxfilesymmetry symmetry,
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
    enum mtxfilesymmetry symmetry,
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
    enum mtxfilesymmetry symmetry,
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
    enum mtxfilesymmetry symmetry,
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
    enum mtxfilesymmetry symmetry,
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
    enum mtxfilesymmetry symmetry,
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
    enum mtxfilefield field,
    enum mtxprecision precision,
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
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = -1;
    mtxfile->size.num_nonzeros = -1;
    mtxfile->precision = precision;
    err = mtxfiledata_alloc(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_rows);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
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
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int num_rows,
    int num_columns,
    int64_t num_nonzeros)
{
    int err;
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

    err = mtxfiledata_alloc(
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilesymmetry symmetry,
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
    enum mtxfilefield field,
    enum mtxprecision precision,
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
    mtxfilecomments_init(&mtxfile->comments);
    mtxfile->size.num_rows = num_rows;
    mtxfile->size.num_columns = -1;
    mtxfile->size.num_nonzeros = num_nonzeros;
    mtxfile->precision = precision;

    err = mtxfiledata_alloc(
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
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;
    return mtxfiledata_set_constant_real_single(
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
    enum mtxprecision precision,
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

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
        precision, num_data_lines);
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
        num_data_lines, 0);
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
    enum mtxprecision precision,
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

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
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
        precision, num_data_lines);
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
        num_data_lines, 0);
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
 * `mtxfile_write()' writes a Matrix Market file to the given path.
 * The file may optionally be compressed by gzip.
 *
 * If `path' is `-', then standard output is used.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
#endif
    }
    return MTX_SUCCESS;
}

/**
 * `mtxfile_fwrite()' writes a Matrix Market file to a stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    err = mtxfileheader_fwrite(&mtxfile->header, f, bytes_written);
    if (err)
        return err;
    err = mtxfilecomments_fputs(&mtxfile->comments, f, bytes_written);
    if (err)
        return err;
    err = mtxfilesize_fwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;
    err = mtxfiledata_fwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines,
        f, fmt, bytes_written);
    if (err)
        return err;
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxfile_gzwrite()' writes a Matrix Market file to a
 * gzip-compressed stream.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
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
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    err = mtxfileheader_gzwrite(&mtxfile->header, f, bytes_written);
    if (err)
        return err;
    err = mtxfilecomments_gzputs(&mtxfile->comments, f, bytes_written);
    if (err)
        return err;
    err = mtxfilesize_gzwrite(
        &mtxfile->size, mtxfile->header.object, mtxfile->header.format,
        f, bytes_written);
    if (err)
        return err;
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;
    err = mtxfiledata_gzwrite(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines,
        f, fmt, bytes_written);
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
        err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
        if (err)
            return err;
        err = mtxfiledata_transpose(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, num_data_lines);
        if (err)
            return err;
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
 * `mtxfile_conjugate_transpose()' tranposes and complex conjugates a
 * Matrix Market file.
 */
int mtxfile_conjugate_transpose(
    struct mtxfile * mtxfile);

/*
 * Sorting
 */

/**
 * `mtxfile_sorting_str()` is a string representing the sorting of a
 * matrix or vector in Matix Market format.
 */
const char * mtxfile_sorting_str(
    enum mtxfile_sorting sorting)
{
    switch (sorting) {
    case mtxfile_unsorted: return "unsorted";
    case mtxfile_sorting_permutation: return "permute";
    case mtxfile_row_major: return "row-major";
    case mtxfile_column_major: return "column-major";
    case mtxfile_morton: return "morton";
    default: return mtxstrerror(MTX_ERR_INVALID_SORTING);
    }
}

/**
 * ‘mtxfile_parse_sorting()’ parses a string containing the ‘sorting’
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
 * On success, ‘mtxfile_parse_sorting()’ returns ‘MTX_SUCCESS’ and
 * ‘sorting’ is set according to the parsed string and ‘bytes_read’ is
 * set to the number of bytes that were consumed by the parser.
 * Otherwise, an error code is returned.
 */
int mtxfile_parse_sorting(
    enum mtxfile_sorting * sorting,
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
        *sorting = mtxfile_sorting_permutation;
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
 * ‘mtxfile_sorting_permutation’, then ‘perm’ must point to an array
 * of ‘size’ integers that specify the sorting permutation. Note that
 * the sorting permutation uses 1-based indexing.
 *
 * For a vector or matrix in coordinate format, the nonzero values are
 * sorted in the specified order. For Matrix Market files in array
 * format, this operation does nothing.
 *
 * ‘size’ is the number of vector or matrix nonzeros to sort.
 *
 * ‘perm’ is ignored if it is ‘NULL’. Otherwise, it must point to an
 * array of ‘size’ 64-bit integers, and it is used to store the
 * permutation of the vector or matrix nonzeros.
 */
int mtxfile_sort(
    struct mtxfile * mtxfile,
    enum mtxfile_sorting sorting,
    int64_t size,
    int64_t * perm)
{
    int64_t num_data_lines;
    int err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;
    if (size < 0 || size > num_data_lines)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (sorting == mtxfile_unsorted) {
        if (!perm)
            return MTX_SUCCESS;
        for (int64_t k = 0; k < size; k++)
            perm[k] = k+1;
        return MTX_SUCCESS;
    } else if (sorting == mtxfile_sorting_permutation) {
        return mtxfiledata_permute(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm);
    } else if (sorting == mtxfile_row_major) {
        return mtxfiledata_sort_row_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm);
    } else if (sorting == mtxfile_column_major) {
        return mtxfiledata_sort_column_major(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm);
    } else if (sorting == mtxfile_morton) {
        return mtxfiledata_sort_morton(
            &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
            mtxfile->header.field, mtxfile->precision, mtxfile->size.num_rows,
            mtxfile->size.num_columns, size, perm);
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
        struct mtxfilesize size;
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
            mtxfilecomments_free(&dst[p].comments);
            return err;
        }

        const int64_t * srcdispls = &data_lines_per_part[data_lines_per_part_ptr[p]];
        err = mtxfiledata_copy_gather(
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
 * data lines is obtained from ‘mtxfilesize_num_data_lines()’). On a
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
    err = mtxfiledata_partition_rows(
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
    err = mtxfileheader_copy(&dst->header, &src->header);
    if (err)
        return err;
    err = mtxfilecomments_copy(&dst->comments, &src->comments);
    if (err)
        return err;
    err = mtxfilesize_copy(&dst->size, &src->size);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    dst->precision = src->precision;

    if (dst->header.format == mtxfile_array) {
        dst->size.num_rows = row_partition->index_sets[part].size;
    } else if (dst->header.format == mtxfile_coordinate) {
        dst->size.num_nonzeros =
            data_lines_per_part_ptr[part+1] - data_lines_per_part_ptr[part];
    } else {
        mtxfilecomments_free(&dst->comments);
        return MTX_ERR_INVALID_MTX_FORMAT;
    }

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &dst->size, dst->header.symmetry, &num_data_lines);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    if (num_data_lines != data_lines_per_part_ptr[part+1]-data_lines_per_part_ptr[part])
    {
        mtxfilecomments_free(&dst->comments);
        errno = EINVAL;
        return MTX_ERR_ERRNO;
    }

    err = mtxfiledata_alloc(
        &dst->data,
        dst->header.object,
        dst->header.format,
        dst->header.field,
        dst->precision, num_data_lines);
    if (err) {
        mtxfilecomments_free(&dst->comments);
        return err;
    }
    err = mtxfiledata_copy(
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
 * Reordering
 */

/**
 * ‘mtxfile_reorder_dims()’ reorders the rows of a vector or the rows
 *  and columns of a matrix in Matrix Market format based on given row
 *  and column permutations.
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
int mtxfile_reorder_dims(
    struct mtxfile * mtxfile,
    const int * row_permutation,
    const int * column_permutation)
{
    int err;
    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;
    err = mtxfiledata_reorder(
        &mtxfile->data, mtxfile->header.object, mtxfile->header.format,
        mtxfile->header.field, mtxfile->precision, num_data_lines, 0,
        mtxfile->size.num_rows, row_permutation,
        mtxfile->size.num_columns, column_permutation);
    if (err)
        return err;
    return MTX_SUCCESS;
}

/**
 * `mtxfile_ordering_str()` is a string representing the ordering
 * of a matrix in Matix Market format.
 */
const char * mtxfile_ordering_str(
    enum mtxfile_ordering ordering)
{
    switch (ordering) {
    case mtxfile_unordered: return "unordered";
    case mtxfile_rcm: return "rcm";
    default: return mtxstrerror(MTX_ERR_INVALID_ORDERING);
    }
}

/**
 * `mtxfile_reorder_rcm()` reorders the rows of a sparse matrix
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
    int * starting_vertex)
{
    int err;
    int num_rows = mtxfile->size.num_rows;
    int num_columns = mtxfile->size.num_columns;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;
    bool square = num_rows == num_columns;
    int num_vertices = square ? num_rows : num_rows + num_columns;

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
        rowptr, colidx);
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
        colptr, rowidx);
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
        err = mtxfile_reorder_dims(mtxfile, rowperm, colperm);
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
 * `mtxfile_reorder()` reorders the rows and columns of a matrix
 * according to the specified algorithm.
 *
 * If successful, this function returns ‘MTX_SUCCESS’, and the rows
 * and columns of ‘mtxfile’ have been reordered according to the
 * specified method. If ‘rowperm’ is not ‘NULL’, then it must point to
 * an array that is large enough to hold one ‘int’ for each row of the
 * matrix. In this case, the array is used to store the permutation
 * for reordering the matrix rows. Similarly, ‘colperm’ is used to
 * store the permutation for reordering the matrix columns.
 */
int mtxfile_reorder(
    struct mtxfile * mtxfile,
    enum mtxfile_ordering ordering,
    int * rowperm,
    int * colperm,
    bool permute,
    int * rcm_starting_vertex)
{
    int err;
    if (ordering == mtxfile_rcm) {
        return mtxfile_reorder_rcm(
            mtxfile, rowperm, colperm, permute, rcm_starting_vertex);
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
    err = mtxfileheader_send(&mtxfile->header, dest, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfilecomments_send(&mtxfile->comments, dest, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfilesize_send(&mtxfile->size, dest, tag, comm, mpierror);
    if (err)
        return err;
    mpierror->mpierrcode = MPI_Send(
        &mtxfile->precision, 1, MPI_INT, dest, tag, comm);
    if (mpierror->mpierrcode)
        return MTX_ERR_MPI;

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err)
        return err;

    err = mtxfiledata_send(
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
    err = mtxfileheader_recv(&mtxfile->header, source, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfilecomments_recv(&mtxfile->comments, source, tag, comm, mpierror);
    if (err)
        return err;
    err = mtxfilesize_recv(&mtxfile->size, source, tag, comm, mpierror);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    mpierror->mpierrcode = MPI_Recv(
        &mtxfile->precision, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
    if (mpierror->mpierrcode) {
        mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI;
    }

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (err) {
        mtxfilecomments_free(&mtxfile->comments);
        return err;
    }

    err = mtxfiledata_alloc(
        &mtxfile->data,
        mtxfile->header.object,
        mtxfile->header.format,
        mtxfile->header.field,
        mtxfile->precision, num_data_lines);
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
        num_data_lines, 0,
        source, tag, comm, mpierror);
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

    err = mtxfileheader_bcast(&mtxfile->header, root, comm, mpierror);
    if (err)
        return err;
    err = mtxfilecomments_bcast(&mtxfile->comments, root, comm, mpierror);
    if (err)
        return err;
    err = mtxfilesize_bcast(&mtxfile->size, root, comm, mpierror);
    if (err) {
        if (rank != root)
            mtxfilecomments_free(&mtxfile->comments);
        return err;
    }
    mpierror->mpierrcode = MPI_Bcast(
        &mtxfile->precision, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        if (rank != root)
            mtxfilecomments_free(&mtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &mtxfile->size, mtxfile->header.symmetry, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
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
            mtxfile->precision, num_data_lines);
    }
    if (mtxmpierror_allreduce(mpierror, err)) {
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
        num_data_lines, 0,
        root, comm, mpierror);
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
        ? mtxfileheader_copy(&recvmtxfile->header, &sendmtxfile->header)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfileheader_bcast(&recvmtxfile->header, root, comm, mpierror);
    if (err)
        return err;

    err = (rank == root)
        ? mtxfilecomments_init(&recvmtxfile->comments)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = (rank == root)
        ? mtxfilecomments_copy(&recvmtxfile->comments, &sendmtxfile->comments)
        : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    err = mtxfilecomments_bcast(&recvmtxfile->comments, root, comm, mpierror);
    if (err) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return err;
    }

    err = mtxfilesize_scatterv(
        &sendmtxfile->size, &recvmtxfile->size,
        recvmtxfile->header.object, recvmtxfile->header.format,
        recvcount, root, comm, mpierror);
    if (err) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return err;
    }

    recvmtxfile->precision = sendmtxfile->precision;
    mpierror->mpierrcode = MPI_Bcast(
        &recvmtxfile->precision, 1, MPI_INT, root, comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    int64_t num_data_lines;
    err = mtxfilesize_num_data_lines(
        &recvmtxfile->size, recvmtxfile->header.symmetry, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = recvcount != num_data_lines ? MTX_ERR_INDEX_OUT_OF_BOUNDS : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_alloc(
        &recvmtxfile->data,
        recvmtxfile->header.object,
        recvmtxfile->header.format,
        recvmtxfile->header.field,
        recvmtxfile->precision,
        recvcount);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxfilecomments_free(&recvmtxfile->comments);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_scatterv(
        &sendmtxfile->data,
        recvmtxfile->header.object, recvmtxfile->header.format,
        recvmtxfile->header.field, recvmtxfile->precision,
        0, sendcounts, displs,
        &recvmtxfile->data, 0, recvcount,
        root, comm, mpierror);
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
