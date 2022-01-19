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
 * Data structures for vectors in coordinate format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/field.h>
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
    const struct mtxvector_coordinate * src);

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
    const struct mtxvector_coordinate * vector,
    struct mtxfile * mtxfile,
    enum mtxfileformat format)
{
    if (format != mtxfile_coordinate)
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
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[k] = xdata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dcopy(x->num_nonzeros, xdata, 1, ydata, 1);
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
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->num_nonzeros, a, xdata, 1);
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
#else
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->num_nonzeros, a, xdata, 1);
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
