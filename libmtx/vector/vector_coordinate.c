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
 * Last modified: 2021-09-20
 *
 * Data structures for vectors in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/field.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/vector_coordinate.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
#endif

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

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
 * `mtxvector_coordinate_free()' frees storage allocated for a vector.
 */
void mtxvector_coordinate_free(
    struct mtxvector_coordinate * vector)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            free(vector->data.real_single);
        } else if (vector->precision == mtx_double) {
            free(vector->data.real_double);
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            free(vector->data.complex_single);
        } else if (vector->precision == mtx_double) {
            free(vector->data.complex_double);
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            free(vector->data.integer_single);
        } else if (vector->precision == mtx_double) {
            free(vector->data.integer_double);
        }
    }
    free(vector->indices);
}

/**
 * `mtxvector_coordinate_alloc_copy()' allocates a copy of a vector
 * without initialising the values.
 */
int mtxvector_coordinate_alloc_copy(
    struct mtxvector_coordinate * dst,
    const struct mtxvector_coordinate * src);

/**
 * `mtxvector_coordinate_init_copy()' allocates a copy of a vector and
 * also copies the values.
 */
int mtxvector_coordinate_init_copy(
    struct mtxvector_coordinate * dst,
    const struct mtxvector_coordinate * src);

/*
 * Vector coordinate formats
 */

/**
 * `mtxvector_coordinate_alloc()' allocates a vector in coordinate
 * format.
 */
int mtxvector_coordinate_alloc(
    struct mtxvector_coordinate * vector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int size,
    int64_t num_nonzeros)
{
    vector->indices = malloc(num_nonzeros * sizeof(int));
    if (!vector->indices)
        return MTX_ERR_ERRNO;
    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            vector->data.real_single =
                malloc(num_nonzeros * sizeof(*vector->data.real_single));
            if (!vector->data.real_single) {
                free(vector->indices);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            vector->data.real_double =
                malloc(num_nonzeros * sizeof(*vector->data.real_double));
            if (!vector->data.real_double) {
                free(vector->indices);
                return MTX_ERR_ERRNO;
            }
        } else {
            free(vector->indices);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            vector->data.complex_single =
                malloc(num_nonzeros * sizeof(*vector->data.complex_single));
            if (!vector->data.complex_single) {
                free(vector->indices);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            vector->data.complex_double =
                malloc(num_nonzeros * sizeof(*vector->data.complex_double));
            if (!vector->data.complex_double) {
                free(vector->indices);
                return MTX_ERR_ERRNO;
            }
        } else {
            free(vector->indices);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            vector->data.integer_single =
                malloc(num_nonzeros * sizeof(*vector->data.integer_single));
            if (!vector->data.integer_single) {
                free(vector->indices);
                return MTX_ERR_ERRNO;
            }
        } else if (precision == mtx_double) {
            vector->data.integer_double =
                malloc(num_nonzeros * sizeof(*vector->data.integer_double));
            if (!vector->data.integer_double) {
                free(vector->indices);
                return MTX_ERR_ERRNO;
            }
        } else {
            free(vector->indices);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_pattern) {
        /* No data needs to be allocated. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    vector->field = field;
    vector->precision = precision;
    vector->size = size;
    vector->num_nonzeros = num_nonzeros;
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_real_single()' allocates and initialises
 * a vector in coordinate format with real, single precision
 * coefficients.
 */
int mtxvector_coordinate_init_real_single(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const float * data)
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_real, mtx_single, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        vector->indices[k] = indices[k];
        vector->data.real_single[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_real_double()' allocates and initialises
 * a vector in coordinate format with real, double precision
 * coefficients.
 */
int mtxvector_coordinate_init_real_double(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const double * data)
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_real, mtx_double, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        vector->indices[k] = indices[k];
        vector->data.real_double[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_complex_single()' allocates and
 * initialises a vector in coordinate format with complex, single
 * precision coefficients.
 */
int mtxvector_coordinate_init_complex_single(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const float (* data)[2])
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_complex, mtx_single, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        vector->indices[k] = indices[k];
        vector->data.complex_single[k][0] = data[k][0];
        vector->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_complex_double()' allocates and
 * initialises a vector in coordinate format with complex, double
 * precision coefficients.
 */
int mtxvector_coordinate_init_complex_double(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const double (* data)[2])
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_complex, mtx_double, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        vector->indices[k] = indices[k];
        vector->data.complex_double[k][0] = data[k][0];
        vector->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_integer_single()' allocates and
 * initialises a vector in coordinate format with integer, single
 * precision coefficients.
 */
int mtxvector_coordinate_init_integer_single(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const int32_t * data)
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_integer, mtx_single, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        vector->indices[k] = indices[k];
        vector->data.integer_single[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_integer_double()' allocates and
 * initialises a vector in coordinate format with integer, double
 * precision coefficients.
 */
int mtxvector_coordinate_init_integer_double(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const int64_t * data)
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_integer, mtx_double, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        vector->indices[k] = indices[k];
        vector->data.integer_double[k] = data[k];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_init_pattern()' allocates and initialises a
 * vector in coordinate format with boolean coefficients.
 */
int mtxvector_coordinate_init_pattern(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices)
{
    int err = mtxvector_coordinate_alloc(
        vector, mtx_field_pattern, mtx_single, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++)
        vector->indices[k] = indices[k];
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxvector_coordinate_from_mtxfile()' converts a vector in Matrix
 * Market format to a vector.
 */
int mtxvector_coordinate_from_mtxfile(
    struct mtxvector_coordinate * vector,
    const struct mtxfile * mtxfile)
{
    int err;
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from array to coordinate. */
    if (mtxfile->header.format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    int size = mtxfile->size.num_rows;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;

    if (mtxfile->header.field == mtxfile_real) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_real, mtxfile->precision, size, num_nonzeros);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_real_single * data =
                mtxfile->data.vector_coordinate_real_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i;
                vector->data.real_single[k] = data[k].a;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_real_double * data =
                mtxfile->data.vector_coordinate_real_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i;
                vector->data.real_double[k] = data[k].a;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_complex, mtxfile->precision, size, num_nonzeros);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_complex_single * data =
                mtxfile->data.vector_coordinate_complex_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i;
                vector->data.complex_single[k][0] = data[k].a[0];
                vector->data.complex_single[k][1] = data[k].a[1];
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_complex_double * data =
                mtxfile->data.vector_coordinate_complex_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i;
                vector->data.complex_double[k][0] = data[k].a[0];
                vector->data.complex_double[k][1] = data[k].a[1];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_integer, mtxfile->precision, size, num_nonzeros);
        if (err)
            return err;
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_integer_single * data =
                mtxfile->data.vector_coordinate_integer_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i;
                vector->data.integer_single[k] = data[k].a;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_integer_double * data =
                mtxfile->data.vector_coordinate_integer_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i;
                vector->data.integer_double[k] = data[k].a;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_pattern, mtx_single, size, num_nonzeros);
        if (err)
            return err;
        const struct mtxfile_vector_coordinate_pattern * data =
            mtxfile->data.vector_coordinate_pattern;
        for (int64_t k = 0; k < num_nonzeros; k++)
            vector->indices[k] = data[k].i;
    } else {
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_to_mtxfile()' converts a vector to a vector
 * in Matrix Market format.
 */
int mtxvector_coordinate_to_mtxfile(
    const struct mtxvector_coordinate * vector,
    struct mtxfile * mtxfile);

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxvector_coordinate_copy()' copies values of a vector, `y = x'.
 */
int mtxvector_coordinate_copy(
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x);

/**
 * `mtxvector_coordinate_sscal()' scales a vector by a single precision floating
 * point scalar, `x = a*x'.
 */
int mtxvector_coordinate_sscal(
    float a,
    struct mtxvector_coordinate * x);

/**
 * `mtxvector_coordinate_dscal()' scales a vector by a double precision floating
 * point scalar, `x = a*x'.
 */
int mtxvector_coordinate_dscal(
    double a,
    struct mtxvector_coordinate * x);

/**
 * `mtxvector_coordinate_saxpy()' adds a vector to another vector multiplied by a
 * single precision floating point value, `y = a*x + y'.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_saxpy(
    float a,
    const struct mtxvector_coordinate * x,
    struct mtxvector_coordinate * y);

/**
 * `mtxvector_coordinate_daxpy()' adds a vector to another vector multiplied by a
 * double precision floating point value, `y = a*x + y'.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_daxpy(
    double a,
    const struct mtxvector_coordinate * x,
    struct mtxvector_coordinate * y);

/**
 * `mtxvector_coordinate_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, `y = a*y + x'.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_saypx(
    float a,
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x);

/**
 * `mtxvector_coordinate_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, `y = a*y + x'.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_daypx(
    double a,
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x);

/**
 * `mtxvector_coordinate_sdot()' computes the Euclidean dot product of
 * two vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_sdot(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    float * dot)
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_sdot(x->num_nonzeros, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->num_nonzeros, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        *dot = x->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_ddot(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    double * dot)
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_sdot(x->num_nonzeros, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->num_nonzeros, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        *dot = x->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_cdotu()' computes the product of the
 * transpose of a complex row vector with another complex row vector
 * in single precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_cdotu(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    float (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_sdot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_zdotu()' computes the product of the
 * transpose of a complex row vector with another complex row vector
 * in double precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_zdotu(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    double (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_ddot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_cdotc()' computes the Euclidean dot product
 * of two complex vectors in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_cdotc(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    float (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_sdot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_zdotc()' computes the Euclidean dot product
 * of two complex vectors in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_zdotc(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    double (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_ddot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_coordinate_snrm2(
    const struct mtxvector_coordinate * x,
    float * nrm2)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_scnrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dznrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_coordinate_dnrm2()' computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_coordinate_dnrm2(
    const struct mtxvector_coordinate * x,
    double * nrm2)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_scnrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dznrm2(x->num_nonzeros, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(*nrm2);
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * `mtxvector_coordinate_send()' sends a vector to another MPI
 * process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to
 * `mtxvector_coordinate_recv()'.
 */
int mtxvector_coordinate_send(
    const struct mtxvector_coordinate * vector,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_coordinate_recv()' receives a vector from another MPI
 * process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxvector_coordinate_send()'.
 */
int mtxvector_coordinate_recv(
    struct mtxvector_coordinate * vector,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_coordinate_bcast()' broadcasts a vector from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxvector_coordinate_bcast()'.
 */
int mtxvector_coordinate_bcast(
    struct mtxvector_coordinate * vector,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);
#endif
