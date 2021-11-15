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
 * Last modified: 2021-09-19
 *
 * Data structures for vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/vector/vector.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtx/precision.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/*
 * Vector types
 */

/**
 * `mtxvector_type_str()' is a string representing the vector type.
 */
const char * mtxvector_type_str(
    enum mtxvector_type type)
{
    switch (type) {
    case mtxvector_auto: return "auto";
    case mtxvector_array: return "array";
    case mtxvector_coordinate: return "coordinate";
    default: return mtx_strerror(MTX_ERR_INVALID_VECTOR_TYPE);
    }
}

/**
 * `mtxvector_type_parse()' parses a string to obtain one of the
 * vector types of `enum mtxvector_type'.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtxvector_type_parse()' returns `MTX_SUCCESS' and
 * `vector_type' is set according to the parsed string and
 * `bytes_read' is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxvector_type_parse(
    enum mtxvector_type * vector_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("auto", t, strlen("auto")) == 0) {
        t += strlen("auto");
        *vector_type = mtxvector_auto;
    } else if (strncmp("array", t, strlen("array")) == 0) {
        t += strlen("array");
        *vector_type = mtxvector_array;
    } else if (strncmp("coordinate", t, strlen("coordinate")) == 0) {
        t += strlen("coordinate");
        *vector_type = mtxvector_coordinate;
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_VECTOR_TYPE;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Memory management
 */

/**
 * `mtxvector_free()' frees storage allocated for a vector.
 */
void mtxvector_free(
    struct mtxvector * vector)
{
    if (vector->type == mtxvector_array) {
        mtxvector_array_free(&vector->storage.array);
    } else if (vector->type == mtxvector_coordinate) {
        mtxvector_coordinate_free(&vector->storage.coordinate);
    }
}

/**
 * `mtxvector_alloc_copy()' allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_alloc_copy(
    struct mtxvector * dst,
    const struct mtxvector * src);

/**
 * `mtxvector_init_copy()' allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_init_copy(
    struct mtxvector * dst,
    const struct mtxvector * src);

/*
 * Vector array formats
 */

/**
 * `mtxvector_alloc_array()' allocates a vector in array
 * format.
 */
int mtxvector_alloc_array(
    struct mtxvector * vector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows)
{
    vector->type = mtxvector_array;
    return mtxvector_array_alloc(
        &vector->storage.array, field, precision, num_rows);
}

/**
 * `mtxvector_init_array_real_single()' allocates and initialises a
 * vector in array format with real, single precision coefficients.
 */
int mtxvector_init_array_real_single(
    struct mtxvector * vector,
    int num_rows,
    const float * data)
{
    vector->type = mtxvector_array;
    return mtxvector_array_init_real_single(
        &vector->storage.array, num_rows, data);
}

/**
 * `mtxvector_init_array_real_double()' allocates and initialises a
 * vector in array format with real, double precision coefficients.
 */
int mtxvector_init_array_real_double(
    struct mtxvector * vector,
    int num_rows,
    const double * data)
{
    vector->type = mtxvector_array;
    return mtxvector_array_init_real_double(
        &vector->storage.array, num_rows, data);
}

/**
 * `mtxvector_init_array_complex_single()' allocates and initialises a
 * vector in array format with complex, single precision coefficients.
 */
int mtxvector_init_array_complex_single(
    struct mtxvector * vector,
    int num_rows,
    const float (* data)[2])
{
    vector->type = mtxvector_array;
    return mtxvector_array_init_complex_single(
        &vector->storage.array, num_rows, data);
}

/**
 * `mtxvector_init_array_complex_double()' allocates and initialises a
 * vector in array format with complex, double precision coefficients.
 */
int mtxvector_init_array_complex_double(
    struct mtxvector * vector,
    int num_rows,
    const double (* data)[2])
{
    vector->type = mtxvector_array;
    return mtxvector_array_init_complex_double(
        &vector->storage.array, num_rows, data);
}

/**
 * `mtxvector_init_array_integer_single()' allocates and initialises a
 * vector in array format with integer, single precision coefficients.
 */
int mtxvector_init_array_integer_single(
    struct mtxvector * vector,
    int num_rows,
    const int32_t * data)
{
    vector->type = mtxvector_array;
    return mtxvector_array_init_integer_single(
        &vector->storage.array, num_rows, data);
}

/**
 * `mtxvector_init_array_integer_double()' allocates and initialises a
 * vector in array format with integer, double precision coefficients.
 */
int mtxvector_init_array_integer_double(
    struct mtxvector * vector,
    int num_rows,
    const int64_t * data)
{
    vector->type = mtxvector_array;
    return mtxvector_array_init_integer_double(
        &vector->storage.array, num_rows, data);
}

/*
 * Vector coordinate formats
 */

/**
 * `mtxvector_alloc_coordinate()' allocates a vector in
 * coordinate format.
 */
int mtxvector_alloc_coordinate(
    struct mtxvector * vector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int num_rows,
    int64_t num_nonzeros)
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_alloc(
        &vector->storage.coordinate, field, precision, num_rows, num_nonzeros);
}

/**
 * `mtxvector_init_coordinate_real_single()' allocates and initialises
 * a vector in coordinate format with real, single precision
 * coefficients.
 */
int mtxvector_init_coordinate_real_single(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices,
    const float * data)
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_real_single(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices, data);
}

/**
 * `mtxvector_init_coordinate_real_double()' allocates and initialises
 * a vector in coordinate format with real, double precision
 * coefficients.
 */
int mtxvector_init_coordinate_real_double(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices,
    const double * data)
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_real_double(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices, data);
}

/**
 * `mtxvector_init_coordinate_complex_single()' allocates and
 * initialises a vector in coordinate format with complex, single
 * precision coefficients.
 */
int mtxvector_init_coordinate_complex_single(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices,
    const float (* data)[2])
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_complex_single(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices, data);
}

/**
 * `mtxvector_init_coordinate_complex_double()' allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxvector_init_coordinate_complex_double(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices,
    const double (* data)[2])
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_complex_double(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices, data);
}

/**
 * `mtxvector_init_coordinate_integer_single()' allocates and
 * initialises a vector in coordinate format with integer, single
 * precision coefficients.
 */
int mtxvector_init_coordinate_integer_single(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices,
    const int32_t * data)
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_integer_single(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices, data);
}

/**
 * `mtxvector_init_coordinate_integer_double()' allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxvector_init_coordinate_integer_double(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices,
    const int64_t * data)
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_integer_double(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices, data);
}

/**
 * `mtxvector_init_coordinate_pattern()' allocates and initialises a
 * vector in coordinate format with integer, double precision
 * coefficients.
 */
int mtxvector_init_coordinate_pattern(
    struct mtxvector * vector,
    int num_rows,
    int64_t num_nonzeros,
    const int * indices)
{
    vector->type = mtxvector_coordinate;
    return mtxvector_coordinate_init_pattern(
        &vector->storage.coordinate, num_rows, num_nonzeros, indices);
}

/*
 * Modifying values
 */

/**
 * `mtxvector_set_constant_real_single()' sets every (nonzero) value
 * of a vector equal to a constant, single precision floating point
 * number.
 */
int mtxvector_set_constant_real_single(
    struct mtxvector * vector,
    float a)
{
    if (vector->type == mtxvector_array) {
        return mtxvector_array_set_constant_real_single(
            &vector->storage.array, a);
    } else if (vector->type == mtxvector_coordinate) {
        return mtxvector_coordinate_set_constant_real_single(
            &vector->storage.coordinate, a);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_set_constant_real_double()' sets every (nonzero) value
 * of a vector equal to a constant, double precision floating point
 * number.
 */
int mtxvector_set_constant_real_double(
    struct mtxvector * mtxvector,
    double a);

/**
 * `mtxvector_set_constant_complex_single()' sets every (nonzero)
 * value of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxvector_set_constant_complex_single(
    struct mtxvector * mtxvector,
    float a[2]);

/**
 * `mtxvector_set_constant_complex_double()' sets every (nonzero)
 * value of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxvector_set_constant_complex_double(
    struct mtxvector * mtxvector,
    double a[2]);

/**
 * `mtxvector_set_constant_integer_single()' sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_single(
    struct mtxvector * mtxvector,
    int32_t a);

/**
 * `mtxvector_set_constant_integer_double()' sets every (nonzero)
 * value of a vector equal to a constant integer.
 */
int mtxvector_set_constant_integer_double(
    struct mtxvector * mtxvector,
    int64_t a);

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxvector_from_mtxfile()' converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_from_mtxfile(
    struct mtxvector * vector,
    const struct mtxfile * mtxfile,
    enum mtxvector_type type)
{
    if (type == mtxvector_auto) {
        if (mtxfile->header.format == mtxfile_array) {
            type = mtxvector_array;
        } else if (mtxfile->header.format == mtxfile_coordinate) {
            type = mtxvector_coordinate;
        } else {
            return MTX_ERR_INVALID_MTX_FORMAT;
        }
    }

    if (type == mtxvector_array) {
        vector->type = mtxvector_array;
        return mtxvector_array_from_mtxfile(
            &vector->storage.array, mtxfile);
    } else if (type == mtxvector_coordinate) {
        vector->type = mtxvector_coordinate;
        return mtxvector_coordinate_from_mtxfile(
            &vector->storage.coordinate, mtxfile);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_to_mtxfile()' converts a vector to a vector in Matrix
 * Market format.
 */
int mtxvector_to_mtxfile(
    const struct mtxvector * vector,
    struct mtxfile * mtxfile)
{
    if (vector->type == mtxvector_array) {
        return mtxvector_array_to_mtxfile(
            &vector->storage.array, mtxfile);
    } else if (vector->type == mtxvector_coordinate) {
        return mtxvector_coordinate_to_mtxfile(
            &vector->storage.coordinate, mtxfile);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/*
 * I/O functions
 */

/**
 * `mtxvector_read()' reads a vector from a Matrix Market file.  The
 * file may optionally be compressed by gzip.
 *
 * The `precision' argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * The `type' argument specifies which format to use for representing
 * the vector.  If `type' is `mtxvector_auto', then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_read(
    struct mtxvector * vector,
    enum mtx_precision precision,
    enum mtxvector_type type,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_read(&mtxfile, precision, path, gzip, lines_read, bytes_read);
    if (err)
        return err;

    err = mtxvector_from_mtxfile(vector, &mtxfile, type);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * `mtxvector_fread()' reads a vector from a stream in Matrix Market
 * format.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * The `type' argument specifies which format to use for representing
 * the vector.  If `type' is `mtxvector_auto', then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_fread(
    struct mtxvector * vector,
    enum mtx_precision precision,
    enum mtxvector_type type,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_fread(
        &mtxfile, precision, f, lines_read, bytes_read, line_max, linebuf);
    if (err)
        return err;

    err = mtxvector_from_mtxfile(vector, &mtxfile, type);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxvector_gzread()' reads a vector from a gzip-compressed stream.
 *
 * `precision' is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * The `type' argument specifies which format to use for representing
 * the vector.  If `type' is `mtxvector_auto', then the underlying
 * vector is stored in array format or coordinate format according to
 * the format of the Matrix Market file.  Otherwise, an attempt is
 * made to convert the vector to the desired type.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxvector_gzread(
    struct mtxvector * vector,
    enum mtx_precision precision,
    enum mtxvector_type type,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_gzread(
        &mtxfile, precision, f, lines_read, bytes_read, line_max, linebuf);
    if (err)
        return err;

    err = mtxvector_from_mtxfile(vector, &mtxfile, type);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
#endif

/**
 * `mtxvector_write()' writes a vector to a Matrix Market file. The
 * file may optionally be compressed by gzip.
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
int mtxvector_write(
    const struct mtxvector * vector,
    const char * path,
    bool gzip,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(vector, &mtxfile);
    if (err)
        return err;
    err = mtxfile_write(
        &mtxfile, path, gzip, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * `mtxvector_fwrite()' writes a vector to a stream.
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
int mtxvector_fwrite(
    const struct mtxvector * vector,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(vector, &mtxfile);
    if (err)
        return err;

    err = mtxfile_fwrite(
        &mtxfile, f, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `mtxvector_gzwrite()' writes a vector to a gzip-compressed stream.
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
int mtxvector_gzwrite(
    const struct mtxvector * vector,
    gzFile f,
    const char * fmt,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(vector, &mtxfile);
    if (err)
        return err;

    err = mtxfile_gzwrite(
        &mtxfile, f, fmt, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxvector_swap()' swaps values of two vectors, simultaneously
 * performing `y <- x' and `x <- y'.
 */
int mtxvector_swap(
    struct mtxvector * x,
    struct mtxvector * y)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_swap(
            &x->storage.array, &y->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_swap(
            &x->storage.coordinate, &y->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_copy()' copies values of a vector, `y = x'.
 */
int mtxvector_copy(
    struct mtxvector * y,
    const struct mtxvector * x)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_copy(
            &y->storage.array, &x->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_copy(
            &y->storage.coordinate, &x->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_sscal()' scales a vector by a single precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_sscal(
    float a,
    struct mtxvector * x)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_sscal(a, &x->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sscal(a, &x->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_dscal()' scales a vector by a double precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_dscal(
    double a,
    struct mtxvector * x)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_dscal(a, &x->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_dscal(a, &x->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_saxpy()' adds a vector to another vector multiplied by a
 * single precision floating point value, ‘y = a*x + y’.
 */
int mtxvector_saxpy(
    float a,
    const struct mtxvector * x,
    struct mtxvector * y)
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_saxpy(
            a, &x->storage.array, &y->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_saxpy(
            a, &x->storage.coordinate, &y->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_daxpy()' adds a vector to another vector multiplied by a
 * double precision floating point value, ‘y = a*x + y’.
 */
int mtxvector_daxpy(
    double a,
    const struct mtxvector * x,
    struct mtxvector * y)
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_daxpy(
            a, &x->storage.array, &y->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_daxpy(
            a, &x->storage.coordinate, &y->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxvector_saypx(
    float a,
    struct mtxvector * y,
    const struct mtxvector * x)
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_saypx(
            a, &y->storage.array, &x->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_saypx(
            a, &y->storage.coordinate, &x->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 */
int mtxvector_daypx(
    double a,
    struct mtxvector * y,
    const struct mtxvector * x)
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_daypx(
            a, &y->storage.array, &x->storage.array);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_daypx(
            a, &y->storage.coordinate, &x->storage.coordinate);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_sdot()' computes the Euclidean dot product of two
 * vectors in single precision floating point.
 */
int mtxvector_sdot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float * dot)
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_sdot(
            &x->storage.array, &y->storage.array, dot);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sdot(
            &x->storage.coordinate, &y->storage.coordinate, dot);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 */
int mtxvector_ddot(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double * dot)
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_ddot(
            &x->storage.array, &y->storage.array, dot);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_ddot(
            &x->storage.coordinate, &y->storage.coordinate, dot);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_cdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxvector_cdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2])
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_cdotu(
            &x->storage.array, &y->storage.array, dot);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_cdotu(
            &x->storage.coordinate, &y->storage.coordinate, dot);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_zdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxvector_zdotu(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2])
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_zdotu(
            &x->storage.array, &y->storage.array, dot);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_zdotu(
            &x->storage.coordinate, &y->storage.coordinate, dot);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_cdotc()' computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 */
int mtxvector_cdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    float (* dot)[2])
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_cdotc(
            &x->storage.array, &y->storage.array, dot);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_cdotc(
            &x->storage.coordinate, &y->storage.coordinate, dot);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_zdotc()' computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 */
int mtxvector_zdotc(
    const struct mtxvector * x,
    const struct mtxvector * y,
    double (* dot)[2])
{
    if (x->type != y->type)
        return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    if (x->type == mtxvector_array) {
        return mtxvector_array_zdotc(
            &x->storage.array, &y->storage.array, dot);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_zdotc(
            &x->storage.coordinate, &y->storage.coordinate, dot);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_snrm2(
    const struct mtxvector * x,
    float * nrm2)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_snrm2(&x->storage.array, nrm2);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_snrm2(&x->storage.coordinate, nrm2);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_dnrm2()' computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_dnrm2(
    const struct mtxvector * x,
    double * nrm2)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_dnrm2(&x->storage.array, nrm2);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_dnrm2(&x->storage.coordinate, nrm2);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_sasum()' computes the sum of absolute values (1-norm) of
 * a vector in single precision floating point.  If the vector is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginaty parts is computed.
 */
int mtxvector_sasum(
    const struct mtxvector * x,
    float * asum)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_sasum(&x->storage.array, asum);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_sasum(&x->storage.coordinate, asum);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_dasum()' computes the sum of absolute values (1-norm) of
 * a vector in double precision floating point.  If the vector is
 * complex-valued, then the sum of the absolute values of the real and
 * imaginaty parts is computed.
 */
int mtxvector_dasum(
    const struct mtxvector * x,
    double * asum)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_dasum(&x->storage.array, asum);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_dasum(&x->storage.coordinate, asum);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}

/**
 * `mtxvector_iamax()' finds the index of the first element having the
 * maximum absolute value.  If the vector is complex-valued, then the
 * index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxvector_iamax(
    const struct mtxvector * x,
    int * iamax)
{
    if (x->type == mtxvector_array) {
        return mtxvector_array_iamax(&x->storage.array, iamax);
    } else if (x->type == mtxvector_coordinate) {
        return mtxvector_coordinate_iamax(&x->storage.coordinate, iamax);
    } else {
        return MTX_ERR_INVALID_VECTOR_TYPE;
    }
}
