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
 * Data structures for vectors in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/field.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/vector_array.h>

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
 * `mtxvector_array_free()' frees storage allocated for a vector.
 */
void mtxvector_array_free(
    struct mtxvector_array * vector)
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
}

/**
 * `mtxvector_array_alloc_copy()' allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_array_alloc_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/**
 * `mtxvector_array_init_copy()' allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_array_init_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/*
 * Vector array formats
 */

/**
 * `mtxvector_array_alloc()' allocates a vector in array format.
 */
int mtxvector_array_alloc(
    struct mtxvector_array * vector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int size)
{
    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            vector->data.real_single =
                malloc(size * sizeof(*vector->data.real_single));
            if (!vector->data.real_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            vector->data.real_double =
                malloc(size * sizeof(*vector->data.real_double));
            if (!vector->data.real_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            vector->data.complex_single =
                malloc(size * sizeof(*vector->data.complex_single));
            if (!vector->data.complex_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            vector->data.complex_double =
                malloc(size * sizeof(*vector->data.complex_double));
            if (!vector->data.complex_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            vector->data.integer_single =
                malloc(size * sizeof(*vector->data.integer_single));
            if (!vector->data.integer_single)
                return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            vector->data.integer_double =
                malloc(size * sizeof(*vector->data.integer_double));
            if (!vector->data.integer_double)
                return MTX_ERR_ERRNO;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    vector->field = field;
    vector->precision = precision;
    vector->size = size;
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_init_real_single()' allocates and initialises a
 * vector in array format with real, single precision coefficients.
 */
int mtxvector_array_init_real_single(
    struct mtxvector_array * vector,
    int size,
    const float * data)
{
    int err = mtxvector_array_alloc(vector, mtx_field_real, mtx_single, size);
    if (err)
        return err;
    for (int64_t k = 0; k < size; k++)
        vector->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_init_real_double()' allocates and initialises a
 * vector in array format with real, double precision coefficients.
 */
int mtxvector_array_init_real_double(
    struct mtxvector_array * vector,
    int size,
    const double * data)
{
    int err = mtxvector_array_alloc(vector, mtx_field_real, mtx_double, size);
    if (err)
        return err;
    for (int64_t k = 0; k < size; k++)
        vector->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_init_complex_single()' allocates and initialises a
 * vector in array format with complex, single precision coefficients.
 */
int mtxvector_array_init_complex_single(
    struct mtxvector_array * vector,
    int size,
    const float (* data)[2])
{
    int err = mtxvector_array_alloc(vector, mtx_field_complex, mtx_single, size);
    if (err)
        return err;
    for (int64_t k = 0; k < size; k++) {
        vector->data.complex_single[k][0] = data[k][0];
        vector->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_init_complex_double()' allocates and initialises a
 * vector in array format with complex, double precision coefficients.
 */
int mtxvector_array_init_complex_double(
    struct mtxvector_array * vector,
    int size,
    const double (* data)[2])
{
    int err = mtxvector_array_alloc(vector, mtx_field_complex, mtx_double, size);
    if (err)
        return err;
    for (int64_t k = 0; k < size; k++) {
        vector->data.complex_double[k][0] = data[k][0];
        vector->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_init_integer_single()' allocates and initialises a
 * vector in array format with integer, single precision coefficients.
 */
int mtxvector_array_init_integer_single(
    struct mtxvector_array * vector,
    int size,
    const int32_t * data)
{
    int err = mtxvector_array_alloc(vector, mtx_field_integer, mtx_single, size);
    if (err)
        return err;
    for (int64_t k = 0; k < size; k++)
        vector->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_init_integer_double()' allocates and initialises a
 * vector in array format with integer, double precision coefficients.
 */
int mtxvector_array_init_integer_double(
    struct mtxvector_array * vector,
    int size,
    const int64_t * data)
{
    int err = mtxvector_array_alloc(vector, mtx_field_integer, mtx_double, size);
    if (err)
        return err;
    for (int64_t k = 0; k < size; k++)
        vector->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * `mtxvector_array_from_mtxfile()' converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_array_from_mtxfile(
    struct mtxvector_array * vector,
    const struct mtxfile * mtxfile)
{
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from coordinate to array. */
    if (mtxfile->header.format != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            return mtxvector_array_init_real_single(
                vector, mtxfile->size.num_rows, mtxfile->data.array_real_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxvector_array_init_real_double(
                vector, mtxfile->size.num_rows, mtxfile->data.array_real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            return mtxvector_array_init_complex_single(
                vector, mtxfile->size.num_rows, mtxfile->data.array_complex_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxvector_array_init_complex_double(
                vector, mtxfile->size.num_rows, mtxfile->data.array_complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            return mtxvector_array_init_integer_single(
                vector, mtxfile->size.num_rows, mtxfile->data.array_integer_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxvector_array_init_integer_double(
                vector, mtxfile->size.num_rows, mtxfile->data.array_integer_double);
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
 * `mtxvector_array_to_mtxfile()' converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_array_to_mtxfile(
    const struct mtxvector_array * vector,
    struct mtxfile * mtxfile);

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxvector_array_copy()' copies values of a vector, `y = x'.
 */
int mtxvector_array_copy(
    struct mtxvector_array * y,
    const struct mtxvector_array * x);

/**
 * `mtxvector_array_sscal()' scales a vector by a single precision floating
 * point scalar, `x = a*x'.
 */
int mtxvector_array_sscal(
    float a,
    struct mtxvector_array * x)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(2*x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(2*x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
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
 * `mtxvector_array_dscal()' scales a vector by a double precision floating
 * point scalar, `x = a*x'.
 */
int mtxvector_array_dscal(
    double a,
    struct mtxvector_array * x)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(2*x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(2*x->size, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
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
 * `mtxvector_array_saxpy()' adds a vector to another vector multiplied by a
 * single precision floating point value, `y = a*x + y'.
 */
int mtxvector_array_saxpy(
    float a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y);

/**
 * `mtxvector_array_daxpy()' adds a vector to another vector multiplied by a
 * double precision floating point value, `y = a*x + y'.
 */
int mtxvector_array_daxpy(
    double a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y);

/**
 * `mtxvector_array_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, `y = a*y + x'.
 */
int mtxvector_array_saypx(
    float a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x);

/**
 * `mtxvector_array_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, `y = a*y + x'.
 */
int mtxvector_array_daypx(
    double a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x);

/**
 * `mtxvector_array_sdot()' computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_sdot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float * dot)
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
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
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_ddot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double * dot)
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
#else
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
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
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_cdotu()' computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_sdot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_zdotu()' computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_ddot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_cdotc()' computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_sdot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_zdotc()' computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2])
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, dot);
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_ddot(x, y, &(*dot)[0]);
    }
    return MTX_SUCCESS;
}

/**
 * `mtxvector_array_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_array_snrm2(
    const struct mtxvector_array * x,
    float * nrm2)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
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
            *nrm2 = cblas_scnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dznrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
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
 * `mtxvector_array_dnrm2()' computes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxvector_array_dnrm2(
    const struct mtxvector_array * x,
    double * nrm2)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
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
            *nrm2 = cblas_scnrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(*nrm2);
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dznrm2(x->size, xdata, 1);
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
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
 * `mtxvector_array_send()' sends a vector to another MPI process.
 *
 * This is analogous to `MPI_Send()' and requires the receiving
 * process to perform a matching call to `mtxvector_array_recv()'.
 */
int mtxvector_array_send(
    const struct mtxvector_array * vector,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_array_recv()' receives a vector from another MPI
 * process.
 *
 * This is analogous to `MPI_Recv()' and requires the sending process
 * to perform a matching call to `mtxvector_array_send()'.
 */
int mtxvector_array_recv(
    struct mtxvector_array * vector,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);

/**
 * `mtxvector_array_bcast()' broadcasts a vector from an MPI root
 * process to other processes in a communicator.
 *
 * This is analogous to `MPI_Bcast()' and requires every process in
 * the communicator to perform matching calls to
 * `mtxvector_array_bcast()'.
 */
int mtxvector_array_bcast(
    struct mtxvector_array * vector,
    int root,
    MPI_Comm comm,
    struct mtxmpierror * mpierror);
#endif
