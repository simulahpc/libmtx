/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-19
 *
 * Data structures for vectors in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/field.h>
#include <libmtx/vector/vector_array.h>

#ifdef LIBMTX_HAVE_BLAS
#include <cblas.h>
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
 * ‘mtxvector_array_free()’ frees storage allocated for a vector.
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
 * ‘mtxvector_array_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_array_alloc_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src)
{
    return mtxvector_array_alloc(dst, src->field, src->precision, src->size);
}

/**
 * ‘mtxvector_array_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_array_init_copy(
    struct mtxvector_array * dst,
    const struct mtxvector_array * src);

/*
 * Vector array formats
 */

/**
 * ‘mtxvector_array_alloc()’ allocates a vector in array format.
 */
int mtxvector_array_alloc(
    struct mtxvector_array * vector,
    enum mtxfield field,
    enum mtxprecision precision,
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
 * ‘mtxvector_array_init_real_single()’ allocates and initialises a
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
 * ‘mtxvector_array_init_real_double()’ allocates and initialises a
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
 * ‘mtxvector_array_init_complex_single()’ allocates and initialises a
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
 * ‘mtxvector_array_init_complex_double()’ allocates and initialises a
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
 * ‘mtxvector_array_init_integer_single()’ allocates and initialises a
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
 * ‘mtxvector_array_init_integer_double()’ allocates and initialises a
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
 * Modifying values
 */

/**
 * ‘mtxvector_array_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_array_set_constant_real_single(
    struct mtxvector_array * vector,
    float a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_array_set_constant_real_double(
    struct mtxvector_array * vector,
    double a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_array_set_constant_complex_single(
    struct mtxvector_array * vector,
    float a[2])
{
    if (vector->field == mtx_field_real ||
        vector->field == mtx_field_integer)
    {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_single[k][0] = a[0];
                vector->data.complex_single[k][1] = a[1];
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_double[k][0] = a[0];
                vector->data.complex_double[k][1] = a[1];
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
 * ‘mtxvector_array_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_array_set_constant_complex_double(
    struct mtxvector_array * vector,
    double a[2])
{
    if (vector->field == mtx_field_real ||
        vector->field == mtx_field_integer)
    {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_single[k][0] = a[0];
                vector->data.complex_single[k][1] = a[1];
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_double[k][0] = a[0];
                vector->data.complex_double[k][1] = a[1];
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
 * ‘mtxvector_array_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_array_set_constant_integer_single(
    struct mtxvector_array * vector,
    int32_t a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_array_set_constant_integer_double(
    struct mtxvector_array * vector,
    int64_t a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->size; k++)
                vector->data.integer_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_array_from_mtxfile()’ converts a vector in Matrix Market
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
 * ‘mtxvector_array_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_array_to_mtxfile(
    const struct mtxvector_array * vector,
    struct mtxfile * mtxfile,
    enum mtxfileformat format)
{
    if (format != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            return mtxfile_init_vector_array_real_single(
                mtxfile, vector->size, vector->data.real_single);
        } else if (vector->precision == mtx_double) {
            return mtxfile_init_vector_array_real_double(
                mtxfile, vector->size, vector->data.real_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            return mtxfile_init_vector_array_complex_single(
                mtxfile, vector->size, vector->data.complex_single);
        } else if (vector->precision == mtx_double) {
            return mtxfile_init_vector_array_complex_double(
                mtxfile, vector->size, vector->data.complex_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            return mtxfile_init_vector_array_integer_single(
                mtxfile, vector->size, vector->data.integer_single);
        } else if (vector->precision == mtx_double) {
            return mtxfile_init_vector_array_integer_double(
                mtxfile, vector->size, vector->data.integer_double);
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_pattern) {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_array_swap()’ swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_swap(
    struct mtxvector_array * x,
    struct mtxvector_array * y)
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sswap(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                float z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
#endif
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dswap(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                double z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_cswap(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                float z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
#endif
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zswap(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                double z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++) {
                int32_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++) {
                int64_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
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
 * ‘mtxvector_array_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_copy(
    struct mtxvector_array * y,
    const struct mtxvector_array * x)
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
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_scopy(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dcopy(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_ccopy(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zcopy(x->size, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_sscal()’ scales a vector by a single precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_array_sscal(
    float a,
    struct mtxvector_array * x,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(2*x->size, a, (float *) xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(2*x->size, a, (double *) xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_dscal()’ scales a vector by a double precision floating
 * point scalar, ‘x = a*x’.
 */
int mtxvector_array_dscal(
    double a,
    struct mtxvector_array * x,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, a, xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(2*x->size, a, (float *) xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(2*x->size, a, (double *) xdata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_saxpy()’ adds a vector to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_saxpy(
    float a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(2*x->size, a, (const float *) xdata, 1, (float *) ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(2*x->size, a, (const double *) xdata, 1, (double *) ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_daxpy()’ adds a vector to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_daxpy(
    double a,
    const struct mtxvector_array * x,
    struct mtxvector_array * y,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(x->size, a, xdata, 1, ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(2*x->size, a, (const float *) xdata, 1, (float *) ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(2*x->size, a, (const double *) xdata, 1, (double *) ydata, 1);
#else
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_saypx(
    float a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_daypx(
    double a,
    struct mtxvector_array * y,
    const struct mtxvector_array * x,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }

    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_sdot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float * dot,
    int64_t * num_flops)
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_ddot(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double * dot,
    int64_t * num_flops)
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->size; k++)
                *dot += xdata[k]*ydata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_cdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2],
    int64_t * num_flops)
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
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            double tmp[2];
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, tmp);
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_zdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotu(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2],
    int64_t * num_flops)
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
            float tmp[2];
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, tmp);
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->size;
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
            if (num_flops) *num_flops += 8*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_cdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    float (* dot)[2],
    int64_t * num_flops)
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
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            double tmp[2];
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, tmp);
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_array_zdotc(
    const struct mtxvector_array * x,
    const struct mtxvector_array * y,
    double (* dot)[2],
    int64_t * num_flops)
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
            float tmp[2];
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, tmp);
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->size; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->size;
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
            if (num_flops) *num_flops += 8*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_array_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_snrm2()’ computes the Euclidean norm of a vector
 * in single precision floating point.
 */
int mtxvector_array_snrm2(
    const struct mtxvector_array * x,
    float * nrm2,
    int64_t * num_flops)
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 4*x->size;
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
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_dnrm2()’ computes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxvector_array_dnrm2(
    const struct mtxvector_array * x,
    double * nrm2,
    int64_t * num_flops)
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
            *nrm2 = sqrt(*nrm2);
#endif
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 2*x->size;
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
            if (num_flops) *num_flops += 4*x->size;
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
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->size; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_array_sasum(
    const struct mtxvector_array * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_sasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabsf(xdata[k]);
#endif
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabs(xdata[k]);
#endif
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_scasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dzasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabs(xdata[k][0]) + fabs(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += abs(xdata[k]);
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += llabs(xdata[k]);
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_array_dasum(
    const struct mtxvector_array * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_sasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabsf(xdata[k]);
#endif
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabs(xdata[k]);
#endif
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_scasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dzasum(x->size, xdata, 1);
#else
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += fabs(xdata[k][0]) + fabs(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += abs(xdata[k]);
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *asum = 0;
            for (int64_t k = 0; k < x->size; k++)
                *asum += llabs(xdata[k]);
            if (num_flops) *num_flops += x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_array_iamax(
    const struct mtxvector_array * x,
    int * iamax)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_isamax(x->size, xdata, 1);
#else
            *iamax = 0;
            float max = x->size > 0 ? fabsf(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabsf(xdata[k])) {
                    max = fabsf(xdata[k]);
                    *iamax = k;
                }
            }
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_idamax(x->size, xdata, 1);
#else
            *iamax = 0;
            double max = x->size > 0 ? fabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabs(xdata[k])) {
                    max = fabs(xdata[k]);
                    *iamax = k;
                }
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_icamax(x->size, xdata, 1);
#else
            *iamax = 0;
            float max = x->size > 0 ? fabsf(xdata[0][0]) + fabsf(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabsf(xdata[k][0]) + fabsf(xdata[k][1])) {
                    max = fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                    *iamax = k;
                }
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_izamax(x->size, xdata, 1);
#else
            *iamax = 0;
            double max = x->size > 0 ? fabs(xdata[0][0]) + fabs(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabs(xdata[k][0]) + fabs(xdata[k][1])) {
                    max = fabs(xdata[k][0]) + fabs(xdata[k][1]);
                    *iamax = k;
                }
            }
#endif
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *iamax = 0;
            int32_t max = x->size > 0 ? abs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < abs(xdata[k])) {
                    max = abs(xdata[k]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *iamax = 0;
            int64_t max = x->size > 0 ? llabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < llabs(xdata[k])) {
                    max = llabs(xdata[k]);
                    *iamax = k;
                }
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}
