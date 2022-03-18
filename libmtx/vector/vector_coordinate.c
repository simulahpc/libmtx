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
 * Last modified: 2022-03-14
 *
 * Data structures for vectors in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>
#include <libmtx/vector/vector.h>
#include <libmtx/vector/vector_coordinate.h>

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
 * ‘mtxvector_coordinate_free()’ frees storage allocated for a vector.
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
 * ‘mtxvector_coordinate_alloc_copy()’ allocates a copy of a vector
 * without initialising the values.
 */
int mtxvector_coordinate_alloc_copy(
    struct mtxvector_coordinate * dst,
    const struct mtxvector_coordinate * src)
{
    return mtxvector_coordinate_alloc(
        dst, src->field, src->precision, src->size, src->num_nonzeros);
}


/**
 * ‘mtxvector_coordinate_init_copy()’ allocates a copy of a vector and
 * also copies the values.
 */
int mtxvector_coordinate_init_copy(
    struct mtxvector_coordinate * dst,
    const struct mtxvector_coordinate * src)
{
    int err = mtxvector_coordinate_alloc_copy(dst, src);
    if (err) return err;
    err = mtxvector_coordinate_copy(dst, src);
    if (err) {
        mtxvector_coordinate_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Vector coordinate formats
 */

/**
 * ‘mtxvector_coordinate_alloc()’ allocates a vector in coordinate
 * format.
 */
int mtxvector_coordinate_alloc(
    struct mtxvector_coordinate * vector,
    enum mtxfield field,
    enum mtxprecision precision,
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
 * ‘mtxvector_coordinate_init_real_single()’ allocates and initialises
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
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
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
 * ‘mtxvector_coordinate_init_real_double()’ allocates and initialises
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
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
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
 * ‘mtxvector_coordinate_init_complex_single()’ allocates and
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
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
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
 * ‘mtxvector_coordinate_init_complex_double()’ allocates and
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
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
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
 * ‘mtxvector_coordinate_init_integer_single()’ allocates and
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
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
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
 * ‘mtxvector_coordinate_init_integer_double()’ allocates and
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
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
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
 * ‘mtxvector_coordinate_init_pattern()’ allocates and initialises a
 * vector in coordinate format with boolean coefficients.
 */
int mtxvector_coordinate_init_pattern(
    struct mtxvector_coordinate * vector,
    int size,
    int64_t num_nonzeros,
    const int * indices)
{
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (indices[k] < 0 || indices[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    err = mtxvector_coordinate_alloc(
        vector, mtx_field_pattern, mtx_single, size, num_nonzeros);
    if (err)
        return err;
    for (int64_t k = 0; k < num_nonzeros; k++)
        vector->indices[k] = indices[k];
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxvector_coordinate_set_constant_real_single()’ sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point number.
 */
int mtxvector_coordinate_set_constant_real_single(
    struct mtxvector_coordinate * vector,
    float a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
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
 * ‘mtxvector_coordinate_set_constant_real_double()’ sets every
 * nonzero value of a vector equal to a constant, double precision
 * floating point number.
 */
int mtxvector_coordinate_set_constant_real_double(
    struct mtxvector_coordinate * vector,
    double a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
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
 * ‘mtxvector_coordinate_set_constant_complex_single()’ sets every
 * nonzero value of a vector equal to a constant, single precision
 * floating point complex number.
 */
int mtxvector_coordinate_set_constant_complex_single(
    struct mtxvector_coordinate * vector,
    float a[2])
{
    if (vector->field == mtx_field_real ||
        vector->field == mtx_field_integer)
    {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_single[k][0] = a[0];
                vector->data.complex_single[k][1] = a[1];
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
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
 * ‘mtxvector_coordinate_set_constant_complex_double()’ sets every
 * nonzero value of a vector equal to a constant, double precision
 * floating point complex number.
 */
int mtxvector_coordinate_set_constant_complex_double(
    struct mtxvector_coordinate * vector,
    double a[2])
{
    if (vector->field == mtx_field_real ||
        vector->field == mtx_field_integer)
    {
        return MTX_ERR_INCOMPATIBLE_FIELD;
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_single[k][0] = a[0];
                vector->data.complex_single[k][1] = a[1];
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
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
 * ‘mtxvector_coordinate_set_constant_integer_single()’ sets every
 * nonzero value of a vector equal to a constant integer.
 */
int mtxvector_coordinate_set_constant_integer_single(
    struct mtxvector_coordinate * vector,
    int32_t a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
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
 * ‘mtxvector_coordinate_set_constant_integer_double()’ sets every
 * nonzero value of a vector equal to a constant integer.
 */
int mtxvector_coordinate_set_constant_integer_double(
    struct mtxvector_coordinate * vector,
    int64_t a)
{
    if (vector->field == mtx_field_real) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.real_double[k] = a;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_single[k][0] = a;
                vector->data.complex_single[k][1] = 0;
            }
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++) {
                vector->data.complex_double[k][0] = a;
                vector->data.complex_double[k][1] = 0;
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        if (vector->precision == mtx_single) {
            for (int k = 0; k < vector->num_nonzeros; k++)
                vector->data.integer_single[k] = a;
        } else if (vector->precision == mtx_double) {
            for (int k = 0; k < vector->num_nonzeros; k++)
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
 * ‘mtxvector_coordinate_from_mtxfile()’ converts a vector in Matrix
 * Market format to a vector in coordinate format.
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

    /* Copy the Matrix Market file and perform assembly to remove any
     * duplicate nonzero entries. */
    struct mtxfile copy;
    err = mtxfile_init_copy(&copy, mtxfile);
    if (err)
        return err;
    err = mtxfile_assemble(&copy, mtxfile_row_major, 0, NULL);
    if (err) {
        mtxfile_free(&copy);
        return err;
    }

    int size = mtxfile->size.num_rows;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;

    if (mtxfile->header.field == mtxfile_real) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_real, mtxfile->precision, size, num_nonzeros);
        if (err) {
            mtxfile_free(&copy);
            return err;
        }
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_real_single * data =
                copy.data.vector_coordinate_real_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i-1;
                vector->data.real_single[k] = data[k].a;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_real_double * data =
                copy.data.vector_coordinate_real_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i-1;
                vector->data.real_double[k] = data[k].a;
            }
        } else {
            mtxfile_free(&copy);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_complex, mtxfile->precision, size, num_nonzeros);
        if (err) {
            mtxfile_free(&copy);
            return err;
        }
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_complex_single * data =
                copy.data.vector_coordinate_complex_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i-1;
                vector->data.complex_single[k][0] = data[k].a[0];
                vector->data.complex_single[k][1] = data[k].a[1];
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_complex_double * data =
                copy.data.vector_coordinate_complex_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i-1;
                vector->data.complex_double[k][0] = data[k].a[0];
                vector->data.complex_double[k][1] = data[k].a[1];
            }
        } else {
            mtxfile_free(&copy);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_integer, mtxfile->precision, size, num_nonzeros);
        if (err) {
            mtxfile_free(&copy);
            return err;
        }
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_integer_single * data =
                copy.data.vector_coordinate_integer_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i-1;
                vector->data.integer_single[k] = data[k].a;
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_integer_double * data =
                copy.data.vector_coordinate_integer_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                vector->indices[k] = data[k].i-1;
                vector->data.integer_double[k] = data[k].a;
            }
        } else {
            mtxfile_free(&copy);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        err = mtxvector_coordinate_alloc(
            vector, mtx_field_pattern, mtx_single, size, num_nonzeros);
        if (err) {
            mtxfile_free(&copy);
            return err;
        }
        const struct mtxfile_vector_coordinate_pattern * data =
            copy.data.vector_coordinate_pattern;
        for (int64_t k = 0; k < num_nonzeros; k++)
            vector->indices[k] = data[k].i-1;
    } else {
        mtxfile_free(&copy);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (vector->indices[k] < 0 ||
            vector->indices[k] >= size)
        {
            mtxvector_coordinate_free(vector);
            mtxfile_free(&copy);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
    }
    mtxfile_free(&copy);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_to_mtxfile()’ converts a vector in coordinate
 * format to a vector in Matrix Market format.
 */
int mtxvector_coordinate_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_coordinate * vector,
    enum mtxfileformat mtxfmt)
{
    if (mtxfmt != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    int err;
    if (vector->field == mtx_field_real) {
        err = mtxfile_alloc_vector_coordinate(
            mtxfile, mtxfile_real, vector->precision,
            vector->size, vector->num_nonzeros);
        if (err)
            return err;
        if (vector->precision == mtx_single) {
            struct mtxfile_vector_coordinate_real_single * data =
                mtxfile->data.vector_coordinate_real_single;
            for (int64_t k = 0; k < vector->num_nonzeros; k++) {
                data[k].i = vector->indices[k]+1;
                data[k].a = vector->data.real_single[k];
            }
        } else if (vector->precision == mtx_double) {
            struct mtxfile_vector_coordinate_real_double * data =
                mtxfile->data.vector_coordinate_real_double;
            for (int64_t k = 0; k < vector->num_nonzeros; k++) {
                data[k].i = vector->indices[k]+1;
                data[k].a = vector->data.real_double[k];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_complex) {
        err = mtxfile_alloc_vector_coordinate(
            mtxfile, mtxfile_complex, vector->precision,
            vector->size, vector->num_nonzeros);
        if (err)
            return err;
        if (vector->precision == mtx_single) {
            struct mtxfile_vector_coordinate_complex_single * data =
                mtxfile->data.vector_coordinate_complex_single;
            for (int64_t k = 0; k < vector->num_nonzeros; k++) {
                data[k].i = vector->indices[k]+1;
                data[k].a[0] = vector->data.complex_single[k][0];
                data[k].a[1] = vector->data.complex_single[k][1];
            }
        } else if (vector->precision == mtx_double) {
            struct mtxfile_vector_coordinate_complex_double * data =
                mtxfile->data.vector_coordinate_complex_double;
            for (int64_t k = 0; k < vector->num_nonzeros; k++) {
                data[k].i = vector->indices[k]+1;
                data[k].a[0] = vector->data.complex_double[k][0];
                data[k].a[1] = vector->data.complex_double[k][1];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_integer) {
        err = mtxfile_alloc_vector_coordinate(
            mtxfile, mtxfile_integer, vector->precision,
            vector->size, vector->num_nonzeros);
        if (err)
            return err;
        if (vector->precision == mtx_single) {
            struct mtxfile_vector_coordinate_integer_single * data =
                mtxfile->data.vector_coordinate_integer_single;
            for (int64_t k = 0; k < vector->num_nonzeros; k++) {
                data[k].i = vector->indices[k]+1;
                data[k].a = vector->data.integer_single[k];
            }
        } else if (vector->precision == mtx_double) {
            struct mtxfile_vector_coordinate_integer_double * data =
                mtxfile->data.vector_coordinate_integer_double;
            for (int64_t k = 0; k < vector->num_nonzeros; k++) {
                data[k].i = vector->indices[k]+1;
                data[k].a = vector->data.integer_double[k];
            }
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (vector->field == mtx_field_pattern) {
        err = mtxfile_alloc_vector_coordinate(
            mtxfile, mtxfile_pattern, mtx_single,
            vector->size, vector->num_nonzeros);
        if (err)
            return err;
        struct mtxfile_vector_coordinate_pattern * data =
            mtxfile->data.vector_coordinate_pattern;
        for (int64_t k = 0; k < vector->num_nonzeros; k++)
            data[k].i = vector->indices[k]+1;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxvector_coordinate_partition()’ partitions a vector into blocks
 * according to the given partitioning.
 *
 * The partition ‘part’ is allowed to be ‘NULL’, in which case a
 * trivial, singleton partition is used to partition the entries of
 * the vector. Otherwise, ‘part’ must partition the entries of the
 * vector ‘src’. That is, ‘part->size’ must be equal to the size of
 * the vector.
 *
 * The argument ‘dsts’ is an array that must have enough storage for
 * ‘P’ values of type ‘struct mtxvector’, where ‘P’ is the number of
 * parts, ‘part->num_parts’.
 *
 * The user is responsible for freeing storage allocated for each
 * vector in the ‘dsts’ array.
 */
int mtxvector_coordinate_partition(
    struct mtxvector * dsts,
    const struct mtxvector_coordinate * src,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile mtxfile;
    err = mtxvector_coordinate_to_mtxfile(&mtxfile, src, mtxfile_coordinate);
    if (err) return err;

    struct mtxfile * dstmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!dstmtxfiles) return MTX_ERR_ERRNO;
    err = mtxfile_partition(dstmtxfiles, &mtxfile, part, NULL);
    if (err) {
        free(dstmtxfiles);
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);

    for (int p = 0; p < num_parts; p++) {
        dsts[p].type = mtxvector_coordinate;
        err = mtxvector_coordinate_from_mtxfile(
            &dsts[p].storage.coordinate, &dstmtxfiles[p]);
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
 * ‘mtxvector_coordinate_join()’ joins together block vectors to form
 * a larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_coordinate_join(
    struct mtxvector_coordinate * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile * srcmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!srcmtxfiles) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        err = mtxvector_to_mtxfile(&srcmtxfiles[p], &srcs[p], mtxfile_coordinate);
        if (err) {
            for (int q = p-1; q >= 0; q--)
                mtxfile_free(&srcmtxfiles[q]);
            free(srcmtxfiles);
            return err;
        }
    }

    struct mtxfile dstmtxfile;
    err = mtxfile_join(&dstmtxfile, srcmtxfiles, part, NULL);
    if (err) {
        for (int p = 0; p < num_parts; p++)
            mtxfile_free(&srcmtxfiles[p]);
        free(srcmtxfiles);
        return err;
    }
    for (int p = 0; p < num_parts; p++)
        mtxfile_free(&srcmtxfiles[p]);
    free(srcmtxfiles);

    err = mtxvector_coordinate_from_mtxfile(dst, &dstmtxfile);
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

/**
 * ‘mtxvector_coordinate_swap()’ swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_swap(
    struct mtxvector_coordinate * x,
    struct mtxvector_coordinate * y)
{
    if (x->field != y->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size || x->num_nonzeros != y->num_nonzeros)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sswap(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                float z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
#endif
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dswap(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
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
            cblas_cswap(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
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
            cblas_zswap(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
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
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                int32_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
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
 * ‘mtxvector_coordinate_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_copy(
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x)
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
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_scopy(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = xdata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dcopy(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
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
            cblas_ccopy(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zcopy(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
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
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = xdata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = xdata[k];
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_sscal()’ scales a vector by a single
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxvector_coordinate_sscal(
    float a,
    struct mtxvector_coordinate * x,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->num_nonzeros, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->num_nonzeros, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(2*x->num_nonzeros, a, (float *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(2*x->num_nonzeros, a, (double *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_dscal()’ scales a vector by a double
 * precision floating point scalar, ‘x = a*x’.
 */
int mtxvector_coordinate_dscal(
    double a,
    struct mtxvector_coordinate * x,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(x->num_nonzeros, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->num_nonzeros, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_sscal(2*x->num_nonzeros, a, (float *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(2*x->num_nonzeros, a, (double *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_coordinate_cscal(
    float a[2],
    struct mtxvector_coordinate * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
        cblas_cscal(x->num_nonzeros, a, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
        double az[2] = {a[0], a[1]};
        cblas_zscal(x->num_nonzeros, az, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_coordinate_zscal(
    double a[2],
    struct mtxvector_coordinate * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
        float ac[2] = {a[0], a[1]};
        cblas_cscal(x->num_nonzeros, ac, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
        cblas_zscal(x->num_nonzeros, a, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_saxpy()’ adds a vector to another one
 * multiplied by a single precision floating point value, ‘y=a*x+y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_saxpy(
    float a,
    const struct mtxvector_coordinate * x,
    struct mtxvector_coordinate * y,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(x->num_nonzeros, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(x->num_nonzeros, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(2*x->num_nonzeros, a, (const float *) xdata, 1, (float *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(2*x->num_nonzeros, a, (const double *) xdata, 1, (double *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_daxpy()’ adds a vector to another vector
 * multiplied by a double precision floating point value, ‘y = a*x+y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_daxpy(
    double a,
    const struct mtxvector_coordinate * x,
    struct mtxvector_coordinate * y,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(x->num_nonzeros, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(x->num_nonzeros, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            cblas_saxpy(2*x->num_nonzeros, a, (const float *) xdata, 1, (float *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_daxpy(2*x->num_nonzeros, a, (const double *) xdata, 1, (double *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_saypx()’ multiplies a vector by a single
 * precision floating point scalar and adds another vector, ‘y=a*y+x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_saypx(
    float a,
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_daypx()’ multiplies a vector by a double
 * precision floating point scalar and adds another vector, ‘y=a*y+x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_daypx(
    double a,
    struct mtxvector_coordinate * y,
    const struct mtxvector_coordinate * x,
    int64_t * num_flops)
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
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        /* Nothing to be done. */
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_sdot()’ computes the Euclidean dot product of
 * two vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_sdot(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    float * dot,
    int64_t * num_flops)
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
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
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
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
 * ‘mtxvector_coordinate_ddot()’ computes the Euclidean dot product of
 * two vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision, size
 * and number of nonzeros.  Furthermore, it is assumed that the
 * locations of the nonzeros is the same for both vectors.
 */
int mtxvector_coordinate_ddot(
    const struct mtxvector_coordinate * x,
    const struct mtxvector_coordinate * y,
    double * dot,
    int64_t * num_flops)
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *dot = cblas_ddot(x->num_nonzeros, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
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
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            *dot = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *dot += xdata[k]*ydata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
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
 * ‘mtxvector_coordinate_cdotu()’ computes the product of the
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
    float (* dot)[2],
    int64_t * num_flops)
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            double tmp[2];
            cblas_zdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, tmp);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_zdotu()’ computes the product of the
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
    double (* dot)[2],
    int64_t * num_flops)
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
            float tmp[2];
            cblas_cdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, tmp);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotu_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_cdotc()’ computes the Euclidean dot product
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
    float (* dot)[2],
    int64_t * num_flops)
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            double tmp[2];
            cblas_zdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, tmp);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_zdotc()’ computes the Euclidean dot product
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
    double (* dot)[2],
    int64_t * num_flops)
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
            float tmp[2];
            cblas_cdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, tmp);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = tmp[0];
            (*dot)[1] = tmp[1];
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_zdotc_sub(x->num_nonzeros, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            (*dot)[0] = (*dot)[1] = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                (*dot)[0] += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                (*dot)[1] += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
#endif
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        (*dot)[1] = 0;
        return mtxvector_coordinate_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_snrm2()’ computes the Euclidean norm of a
 * vector in single precision floating point.
 */
int mtxvector_coordinate_snrm2(
    const struct mtxvector_coordinate * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_scnrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(*nrm2);
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dznrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(*nrm2);
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrtf(*nrm2);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        *nrm2 = sqrtf(x->num_nonzeros);
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_dnrm2()’ computes the Euclidean norm of a
 * vector in double precision floating point.
 */
int mtxvector_coordinate_dnrm2(
    const struct mtxvector_coordinate * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_snrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dnrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_scnrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(*nrm2);
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *nrm2 = cblas_dznrm2(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(*nrm2);
#endif
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *nrm2 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *nrm2 += xdata[k]*xdata[k];
            *nrm2 = sqrt(*nrm2);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        *nrm2 = sqrtf(x->num_nonzeros);
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_coordinate_sasum(
    const struct mtxvector_coordinate * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_sasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabsf(xdata[k]);
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabs(xdata[k]);
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_scasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dzasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabs(xdata[k][0]) + fabs(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += abs(xdata[k]);
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += llabs(xdata[k]);
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        *asum = x->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_coordinate_dasum(
    const struct mtxvector_coordinate * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_sasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabsf(xdata[k]);
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabs(xdata[k]);
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_scasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *asum = cblas_dzasum(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += fabs(xdata[k][0]) + fabs(xdata[k][1]);
#endif
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += abs(xdata[k]);
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *asum = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                *asum += llabs(xdata[k]);
            if (num_flops) *num_flops += x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        *asum = x->num_nonzeros;
    } else {
        return MTX_ERR_INVALID_FIELD;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_coordinate_iamax(
    const struct mtxvector_coordinate * x,
    int * iamax)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_isamax(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *iamax = 0;
            float max = x->num_nonzeros > 0 ? fabsf(xdata[0]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < fabsf(xdata[k])) {
                    max = fabsf(xdata[k]);
                    *iamax = k;
                }
            }
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_idamax(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *iamax = 0;
            double max = x->num_nonzeros > 0 ? fabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
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
            *iamax = cblas_icamax(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *iamax = 0;
            float max = x->num_nonzeros > 0 ? fabsf(xdata[0][0]) + fabsf(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < fabsf(xdata[k][0]) + fabsf(xdata[k][1])) {
                    max = fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                    *iamax = k;
                }
            }
#endif
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
            *iamax = cblas_izamax(x->num_nonzeros, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            *iamax = 0;
            double max = x->num_nonzeros > 0 ? fabs(xdata[0][0]) + fabs(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
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
            int32_t max = x->num_nonzeros > 0 ? abs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < abs(xdata[k])) {
                    max = abs(xdata[k]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            *iamax = 0;
            int64_t max = x->num_nonzeros > 0 ? llabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
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

/*
 * Sorting
 */

/**
 * ‘mtxvector_coordinate_permute()’ permutes the elements of a vector
 * according to a given permutation.
 *
 * The array ‘perm’ should be an array of length ‘size’ that stores a
 * permutation of the integers ‘0,1,...,N-1’, where ‘N’ is the number
 * of vector elements.
 *
 * After permuting, the 1st vector element of the original vector is
 * now located at position ‘perm[0]’ in the sorted vector ‘x’, the 2nd
 * element is now at position ‘perm[1]’, and so on.
 */
int mtxvector_coordinate_permute(
    struct mtxvector_coordinate * x,
    int64_t offset,
    int64_t size,
    int64_t * perm)
{
    if (offset + size > x->size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    for (int64_t k = 0; k < size; k++) {
        if (perm[k] < 0 || perm[k] >= x->size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* 1. Copy the original, unsorted data. */
    struct mtxvector_coordinate y;
    int err = mtxvector_coordinate_init_copy(&y, x);
    if (err) return err;

    /* 2. Permute the data. */
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * dst = x->data.real_single;
            const float * src = y.data.real_single;
            for (int64_t k = 0; k < size; k++) {
                x->indices[perm[k]] = y.indices[offset+k];
                dst[perm[k]] = src[offset+k];
            }
        } else if (x->precision == mtx_double) {
            double * dst = x->data.real_double;
            const double * src = y.data.real_double;
            for (int64_t k = 0; k < size; k++) {
                x->indices[perm[k]] = y.indices[offset+k];
                dst[perm[k]] = src[offset+k];
            }
        } else {
            mtxvector_coordinate_free(&y);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* dst)[2] = x->data.complex_single;
            const float (* src)[2] = y.data.complex_single;
            for (int64_t k = 0; k < size; k++) {
                x->indices[perm[k]] = y.indices[offset+k];
                dst[perm[k]][0] = src[offset+k][0];
                dst[perm[k]][1] = src[offset+k][1];
            }
        } else if (x->precision == mtx_double) {
            double (* dst)[2] = x->data.complex_double;
            const double (* src)[2] = y.data.complex_double;
            for (int64_t k = 0; k < size; k++) {
                x->indices[perm[k]] = y.indices[offset+k];
                dst[perm[k]][0] = src[offset+k][0];
                dst[perm[k]][1] = src[offset+k][1];
            }
        } else {
            mtxvector_coordinate_free(&y);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * dst = x->data.integer_single;
            const int32_t * src = y.data.integer_single;
            for (int64_t k = 0; k < size; k++) {
                x->indices[perm[k]] = y.indices[offset+k];
                dst[perm[k]] = src[offset+k];
            }
        } else if (x->precision == mtx_double) {
            int64_t * dst = x->data.integer_double;
            const int64_t * src = y.data.integer_double;
            for (int64_t k = 0; k < size; k++) {
                x->indices[perm[k]] = y.indices[offset+k];
                dst[perm[k]] = src[offset+k];
            }
        } else {
            mtxvector_coordinate_free(&y);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_pattern) {
        for (int64_t k = 0; k < size; k++)
            x->indices[perm[k]] = y.indices[offset+k];
    } else {
        mtxvector_coordinate_free(&y);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    mtxvector_coordinate_free(&y);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_coordinate_sort()’ sorts elements of a vector by the
 * given keys.
 *
 * The array ‘keys’ must be an array of length ‘size’ that stores a
 * 64-bit unsigned integer sorting key that is used to define the
 * order in which to sort the vector elements..
 *
 * If it is not ‘NULL’, then ‘perm’ must point to an array of length
 * ‘size’, which is then used to store the sorting permutation. That
 * is, ‘perm’ is a permutation of the integers ‘0,1,...,N-1’, where
 * ‘N’ is the number of vector elements, such that the 1st vector
 * element in the original vector is now located at position ‘perm[0]’
 * in the sorted vector ‘x’, the 2nd element is now at position
 * ‘perm[1]’, and so on.
 */
int mtxvector_coordinate_sort(
    struct mtxvector_coordinate * x,
    int64_t size,
    uint64_t * keys,
    int64_t * perm)
{
    /* 1. Sort the keys and obtain a sorting permutation. */
    bool alloc_perm = !perm;
    if (alloc_perm) {
        perm = malloc(size * sizeof(int64_t));
        if (!perm)
            return MTX_ERR_ERRNO;
    }
    int err = radix_sort_uint64(size, keys, perm);
    if (err) {
        if (alloc_perm) free(perm);
        return err;
    }

    /* 2. Sort data according to the sorting permutation. */
    err = mtxvector_coordinate_permute(x, 0, size, perm);
    if (err) {
        if (alloc_perm) free(perm);
        return err;
    }
    if (alloc_perm) free(perm);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_coordinate_send()’ sends Matrix Market data lines to
 * another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to
 * ‘mtxvector_coordinate_recv()’.
 */
int mtxvector_coordinate_send(
    const struct mtxvector_coordinate * data,
    int64_t size,
    int64_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_recv()’ receives Matrix Market data lines
 * from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_coordinate_send()’.
 */
int mtxvector_coordinate_recv(
    struct mtxvector_coordinate * data,
    int64_t size,
    int64_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_bcast()’ broadcasts Matrix Market data lines
 * from an MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_coordinate_bcast()’.
 */
int mtxvector_coordinate_bcast(
    struct mtxvector_coordinate * data,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_gatherv()’ gathers Matrix Market data lines
 * onto an MPI root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_coordinate_gatherv()’.
 */
int mtxvector_coordinate_gatherv(
    const struct mtxvector_coordinate * sendbuf,
    int64_t sendoffset,
    int sendcount,
    struct mtxvector_coordinate * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_scatterv()’ scatters Matrix Market data lines
 * from an MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_coordinate_scatterv()’.
 */
int mtxvector_coordinate_scatterv(
    const struct mtxvector_coordinate * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    struct mtxvector_coordinate * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_coordinate_alltoallv()’ performs an all-to-all exchange
 * of Matrix Market data lines between MPI processes in a
 * communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxvector_coordinate_alltoallv()’.
 */
int mtxvector_coordinate_alltoallv(
    const struct mtxvector_coordinate * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    struct mtxvector_coordinate * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    return MTX_SUCCESS;
}
#endif
