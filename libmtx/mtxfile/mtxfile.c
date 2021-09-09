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
    const float * data);

/**
 * `mtxfile_init_vector_array_real_double()' allocates and initialises
 * a vector in array format with real, double precision coefficients.
 */
int mtxfile_init_vector_array_real_double(
    struct mtxfile * mtxfile,
    int num_rows,
    const double * data)
{
    int err;
    err = mtxfile_alloc_vector_array(mtxfile, mtxfile_real, mtx_double, num_rows);
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
    const float (* data)[2]);

/**
 * `mtxfile_init_vector_array_complex_double()' allocates and
 * initialises a vector in array format with complex, double precision
 * coefficients.
 */
int mtxfile_init_vector_array_complex_double(
    struct mtxfile * mtxfile,
    int num_rows,
    const double (* data)[2]);

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
    int err;
    err = mtxfile_alloc_matrix_coordinate(
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
    const struct mtxfile_matrix_coordinate_complex_single * data);

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
    const struct mtxfile_matrix_coordinate_complex_double * data);

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
    const struct mtxfile_matrix_coordinate_integer_single * data);

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
    const struct mtxfile_matrix_coordinate_integer_double * data);

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
    const struct mtxfile_matrix_coordinate_pattern * data);

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
    int64_t num_nonzeros);

/**
 * `mtxfile_alloc_vector_coordinate()' allocates a vector in
 * coordinate format.
 */
int mtxfile_alloc_vector_coordinate(
    struct mtxfile * mtxfile,
    enum mtxfile_field field,
    enum mtx_precision precision,
    int num_rows,
    int64_t num_nonzeros);

/**
 * `mtxfile_init_vector_coordinate_real_single()' allocates and initialises
 * a vector in coordinate format with real, single precision coefficients.
 */
int mtxfile_init_vector_coordinate_real_single(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_single * data);

/**
 * `mtxfile_init_vector_coordinate_real_double()' allocates and initialises
 * a vector in coordinate format with real, double precision coefficients.
 */
int mtxfile_init_vector_coordinate_real_double(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_real_double * data);

/**
 * `mtxfile_init_vector_coordinate_complex_single()' allocates and
 * initialises a vector in coordinate format with complex, single precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_complex_single(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_single * data);

/**
 * `mtxfile_init_vector_coordinate_complex_double()' allocates and
 * initialises a vector in coordinate format with complex, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_complex_double(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_complex_double * data);

/**
 * `mtxfile_init_vector_coordinate_integer_single()' allocates and
 * initialises a vector in coordinate format with integer, single precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_integer_single(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_single * data);

/**
 * `mtxfile_init_vector_coordinate_integer_double()' allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_integer_double(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_integer_double * data);

/**
 * `mtxfile_init_vector_coordinate_pattern()' allocates and
 * initialises a vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxfile_init_vector_coordinate_pattern(
    struct mtxfile * mtxfile,
    int num_rows,
    int64_t num_nonzeros,
    const struct mtxfile_vector_coordinate_pattern * data);

/*
 * I/O functions
 */

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
    FILE * f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtx_precision precision)
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
    gzFile f,
    int * lines_read,
    int * bytes_read,
    size_t line_max,
    char * linebuf,
    enum mtx_precision precision)
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
#endif
