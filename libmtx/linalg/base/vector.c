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
 * Last modified: 2022-10-03
 *
 * Data structures and routines for basic dense vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/local/vector.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * vector properties
 */

/**
 * ‘mtxbasevector_field()’ gets the field of a vector.
 */
enum mtxfield mtxbasevector_field(const struct mtxbasevector * x)
{
    return x->field;
}

/**
 * ‘mtxbasevector_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxbasevector_precision(const struct mtxbasevector * x)
{
    return x->precision;
}

/**
 * ‘mtxbasevector_size()’ gets the size of a vector.
 */
int64_t mtxbasevector_size(const struct mtxbasevector * x)
{
    return x->size;
}

/**
 * ‘mtxbasevector_num_nonzeros()’ gets the number of explicitly
 * stored vector entries.
 */
int64_t mtxbasevector_num_nonzeros(const struct mtxbasevector * x)
{
    return x->num_nonzeros;
}

/**
 * ‘mtxbasevector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxbasevector_idx(const struct mtxbasevector * x)
{
    return x->idx;
}

/*
 * memory management
 */

/**
 * ‘mtxbasevector_free()’ frees storage allocated for a vector.
 */
void mtxbasevector_free(
    struct mtxbasevector * x)
{
    free(x->idx);
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            free(x->data.real_single);
        } else if (x->precision == mtx_double) {
            free(x->data.real_double);
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            free(x->data.complex_single);
        } else if (x->precision == mtx_double) {
            free(x->data.complex_double);
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            free(x->data.integer_single);
        } else if (x->precision == mtx_double) {
            free(x->data.integer_double);
        }
    }
}

/**
 * ‘mtxbasevector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxbasevector_alloc_copy(
    struct mtxbasevector * dst,
    const struct mtxbasevector * src)
{
    if (src->idx) {
        return mtxbasevector_alloc_packed(
            dst, src->field, src->precision, src->size,
            src->num_nonzeros, src->idx);
    } else {
        return mtxbasevector_alloc(
            dst, src->field, src->precision, src->size);
    }
}

/**
 * ‘mtxbasevector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxbasevector_init_copy(
    struct mtxbasevector * dst,
    const struct mtxbasevector * src)
{
    int err = mtxbasevector_alloc_copy(dst, src);
    if (err) return err;
    err = mtxbasevector_copy(dst, src);
    if (err) {
        mtxbasevector_free(dst);
        return err;
    }
    if (src->idx) {
        for (int64_t k = 0; k < src->num_nonzeros; k++)
            dst->idx[k] = src->idx[k];
    }
    return MTX_SUCCESS;
}

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxbasevector_alloc()’ allocates a vector.
 */
int mtxbasevector_alloc(
    struct mtxbasevector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    if (field == mtx_field_real) {
        if (precision == mtx_single) {
            x->data.real_single = malloc(size * sizeof(*x->data.real_single));
            if (!x->data.real_single) return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            x->data.real_double = malloc(size * sizeof(*x->data.real_double));
            if (!x->data.real_double) return MTX_ERR_ERRNO;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_complex) {
        if (precision == mtx_single) {
            x->data.complex_single = malloc(size * sizeof(*x->data.complex_single));
            if (!x->data.complex_single) return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            x->data.complex_double = malloc(size * sizeof(*x->data.complex_double));
            if (!x->data.complex_double) return MTX_ERR_ERRNO;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_integer) {
        if (precision == mtx_single) {
            x->data.integer_single = malloc(size * sizeof(*x->data.integer_single));
            if (!x->data.integer_single) return MTX_ERR_ERRNO;
        } else if (precision == mtx_double) {
            x->data.integer_double = malloc(size * sizeof(*x->data.integer_double));
            if (!x->data.integer_double) return MTX_ERR_ERRNO;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_pattern) {
        x->data.pattern = NULL;
    } else { return MTX_ERR_INVALID_FIELD; }
    x->field = field;
    x->precision = precision;
    x->size = size;
    x->num_nonzeros = size;
    x->idx = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxbasevector_init_real_single(
    struct mtxbasevector * x,
    int64_t size,
    const float * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxbasevector_init_real_double(
    struct mtxbasevector * x,
    int64_t size,
    const double * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxbasevector_init_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    const float (* data)[2])
{
    int err = mtxbasevector_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++) {
        x->data.complex_single[k][0] = data[k][0];
        x->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxbasevector_init_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    const double (* data)[2])
{
    int err = mtxbasevector_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++) {
        x->data.complex_double[k][0] = data[k][0];
        x->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxbasevector_init_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    const int32_t * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxbasevector_init_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    const int64_t * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxbasevector_init_pattern(
    struct mtxbasevector * x,
    int64_t size)
{
    return mtxbasevector_alloc(x, mtx_field_pattern, mtx_single, size);
}

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxbasevector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxbasevector_init_strided_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const float * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    err = mtxbasevector_set_real_single(x, size, stride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxbasevector_init_strided_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const double * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    err = mtxbasevector_set_real_double(x, size, stride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxbasevector_init_strided_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2])
{
    int err = mtxbasevector_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    err = mtxbasevector_set_complex_single(x, size, stride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxbasevector_init_strided_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2])
{
    int err = mtxbasevector_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    err = mtxbasevector_set_complex_double(x, size, stride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxbasevector_init_strided_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    err = mtxbasevector_set_integer_single(x, size, stride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxbasevector_init_strided_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data)
{
    int err = mtxbasevector_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    err = mtxbasevector_set_integer_double(x, size, stride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxbasevector_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxbasevector_alloc_packed(
    struct mtxbasevector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    int err = mtxbasevector_alloc(x, field, precision, size);
    if (err) return err;
    x->num_nonzeros = num_nonzeros;
    x->idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->idx) { mtxbasevector_free(x); return MTX_ERR_ERRNO; }
    if (idx) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (idx[k] < 0 || idx[k] >= size) {
                free(x->idx); mtxbasevector_free(x);
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
            x->idx[k] = idx[k];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxbasevector_init_packed_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data)
{
    int err = mtxbasevector_alloc_packed(
        x, mtx_field_real, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    for (int64_t k = 0; k < num_nonzeros; k++)
        x->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxbasevector_init_packed_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data)
{
    int err = mtxbasevector_alloc_packed(
        x, mtx_field_real, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    for (int64_t k = 0; k < num_nonzeros; k++)
        x->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxbasevector_init_packed_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2])
{
    int err = mtxbasevector_alloc_packed(
        x, mtx_field_complex, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        x->data.complex_single[k][0] = data[k][0];
        x->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxbasevector_init_packed_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2])
{
    int err = mtxbasevector_alloc_packed(
        x, mtx_field_complex, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        x->data.complex_double[k][0] = data[k][0];
        x->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxbasevector_init_packed_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data)
{
    int err = mtxbasevector_alloc_packed(
        x, mtx_field_integer, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    for (int64_t k = 0; k < num_nonzeros; k++)
        x->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxbasevector_init_packed_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data)
{
    int err = mtxbasevector_alloc_packed(
        x, mtx_field_integer, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    for (int64_t k = 0; k < num_nonzeros; k++)
        x->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxbasevector_init_packed_pattern(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    return mtxbasevector_alloc_packed(
        x, mtx_field_pattern, mtx_single, size, num_nonzeros, idx);
}

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxbasevector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxbasevector_alloc_packed_strided(
    struct mtxbasevector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    int err = mtxbasevector_alloc(x, field, precision, size);
    if (err) return err;
    if (idx) {
        x->num_nonzeros = num_nonzeros;
        x->idx = malloc(num_nonzeros * sizeof(int64_t));
        if (!x->idx) { mtxbasevector_free(x); return MTX_ERR_ERRNO; }
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t idxk = (*(int64_t *) ((unsigned char *) idx + k*idxstride)) - idxbase;
            if (idxk < 0 || idxk >= size) {
                free(x->idx); mtxbasevector_free(x);
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
            x->idx[k] = idxk;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxbasevector_init_packed_strided_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data)
{
    int err = mtxbasevector_alloc_packed_strided(
        x, mtx_field_real, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxbasevector_set_real_single(x, num_nonzeros, datastride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxbasevector_init_packed_strided_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data)
{
    int err = mtxbasevector_alloc_packed_strided(
        x, mtx_field_real, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxbasevector_set_real_double(x, num_nonzeros, datastride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxbasevector_init_packed_strided_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2])
{
    int err = mtxbasevector_alloc_packed_strided(
        x, mtx_field_complex, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxbasevector_set_complex_single(x, num_nonzeros, datastride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxbasevector_init_packed_strided_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2])
{
    int err = mtxbasevector_alloc_packed_strided(
        x, mtx_field_complex, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxbasevector_set_complex_double(x, num_nonzeros, datastride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxbasevector_init_packed_strided_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data)
{
    int err = mtxbasevector_alloc_packed_strided(
        x, mtx_field_integer, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxbasevector_set_integer_single(x, num_nonzeros, datastride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxbasevector_init_packed_strided_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data)
{
    int err = mtxbasevector_alloc_packed_strided(
        x, mtx_field_integer, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxbasevector_set_integer_double(x, num_nonzeros, datastride, data);
    if (err) { mtxbasevector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxbasevector_init_packed_strided_pattern(
    struct mtxbasevector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    return mtxbasevector_alloc_packed_strided(
        x, mtx_field_pattern, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
}

/*
 * accessing values
 */

/**
 * ‘mtxbasevector_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int mtxbasevector_get_real_single(
    const struct mtxbasevector * x,
    int64_t size,
    int stride,
    float * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    float * b = x->data.real_single;
    for (int64_t i = 0; i < x->num_nonzeros; i++)
        *(float *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int mtxbasevector_get_real_double(
    const struct mtxbasevector * x,
    int64_t size,
    int stride,
    double * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    double * b = x->data.real_double;
    for (int64_t i = 0; i < x->num_nonzeros; i++)
        *(double *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int mtxbasevector_get_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    float (* a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    float (* b)[2] = x->data.complex_single;
    for (int64_t i = 0; i < x->num_nonzeros; i++) {
        (*(float (*)[2])((char *) a + i*stride))[0] = b[i][0];
        (*(float (*)[2])((char *) a + i*stride))[1] = b[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int mtxbasevector_get_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    double (* a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    double (* b)[2] = x->data.complex_double;
    for (int64_t i = 0; i < x->num_nonzeros; i++) {
        (*(double (*)[2])((char *) a + i*stride))[0] = b[i][0];
        (*(double (*)[2])((char *) a + i*stride))[1] = b[i][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int mtxbasevector_get_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    int32_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int32_t * b = x->data.integer_single;
    for (int64_t i = 0; i < x->num_nonzeros; i++)
        *(int32_t *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of nonzero vector elements.
 */
int mtxbasevector_get_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    int64_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t * b = x->data.integer_double;
    for (int64_t i = 0; i < x->num_nonzeros; i++)
        *(int64_t *)((char *) a + i*stride) = b[i];
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxbasevector_setzero()’ sets every value of a vector to zero.
 */
int mtxbasevector_setzero(
    struct mtxbasevector * x)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_single[k] = 0;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_double[k] = 0;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = 0;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = 0;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_single[k] = 0;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_double[k] = 0;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxbasevector_set_constant_real_single(
    struct mtxbasevector * x,
    float a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxbasevector_set_constant_real_double(
    struct mtxbasevector * x,
    double a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxbasevector_set_constant_complex_single(
    struct mtxbasevector * x,
    float a[2])
{
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxbasevector_set_constant_complex_double(
    struct mtxbasevector * x,
    double a[2])
{
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxbasevector_set_constant_integer_single(
    struct mtxbasevector * x,
    int32_t a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxbasevector_set_constant_integer_double(
    struct mtxbasevector * x,
    int64_t a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->num_nonzeros; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxbasevector_set_real_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const float * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    float * b = x->data.real_single;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const float *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxbasevector_set_real_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const double * a)
{
    if (x->field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    double * b = x->data.real_double;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const double *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxbasevector_set_complex_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    float (*b)[2] = x->data.complex_single;
    for (int64_t i = 0; i < size; i++) {
        b[i][0] = (*(const float (*)[2])((const char *) a + i*stride))[0];
        b[i][1] = (*(const float (*)[2])((const char *) a + i*stride))[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxbasevector_set_complex_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    if (x->field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    double (*b)[2] = x->data.complex_double;
    for (int64_t i = 0; i < size; i++) {
        b[i][0] = (*(const double (*)[2])((const char *) a + i*stride))[0];
        b[i][1] = (*(const double (*)[2])((const char *) a + i*stride))[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxbasevector_set_integer_single(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const int32_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int32_t * b = x->data.integer_single;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const int32_t *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxbasevector_set_integer_double(
    struct mtxbasevector * x,
    int64_t size,
    int stride,
    const int64_t * a)
{
    if (x->field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int64_t * b = x->data.integer_double;
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const int64_t *)((const char *) a + i*stride);
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxbasevector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxbasevector_from_mtxfile(
    struct mtxbasevector * x,
    const struct mtxfile * mtxfile)
{
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    if (mtxfile->header.format == mtxfile_array) {
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                return mtxbasevector_init_real_single(
                    x, mtxfile->size.num_rows, mtxfile->data.array_real_single);
            } else if (mtxfile->precision == mtx_double) {
                return mtxbasevector_init_real_double(
                    x, mtxfile->size.num_rows, mtxfile->data.array_real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                return mtxbasevector_init_complex_single(
                    x, mtxfile->size.num_rows, mtxfile->data.array_complex_single);
            } else if (mtxfile->precision == mtx_double) {
                return mtxbasevector_init_complex_double(
                    x, mtxfile->size.num_rows, mtxfile->data.array_complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                return mtxbasevector_init_integer_single(
                    x, mtxfile->size.num_rows, mtxfile->data.array_integer_single);
            } else if (mtxfile->precision == mtx_double) {
                return mtxbasevector_init_integer_double(
                    x, mtxfile->size.num_rows, mtxfile->data.array_integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_pattern) {
                return mtxbasevector_init_pattern(x, mtxfile->size.num_rows);
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxfile->header.format == mtxfile_coordinate) {
        int64_t size = mtxfile->size.num_rows;
        int64_t num_nonzeros = mtxfile->size.num_nonzeros;
        if (mtxfile->header.field == mtxfile_real) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_vector_coordinate_real_single * data =
                    mtxfile->data.vector_coordinate_real_single;
                return mtxbasevector_init_packed_strided_real_single(
                    x, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a);
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_vector_coordinate_real_double * data =
                    mtxfile->data.vector_coordinate_real_double;
                return mtxbasevector_init_packed_strided_real_double(
                    x, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_complex) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_vector_coordinate_complex_single * data =
                    mtxfile->data.vector_coordinate_complex_single;
                return mtxbasevector_init_packed_strided_complex_single(
                    x, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a);
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_vector_coordinate_complex_double * data =
                    mtxfile->data.vector_coordinate_complex_double;
                return mtxbasevector_init_packed_strided_complex_double(
                    x, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_integer) {
            if (mtxfile->precision == mtx_single) {
                const struct mtxfile_vector_coordinate_integer_single * data =
                    mtxfile->data.vector_coordinate_integer_single;
                return mtxbasevector_init_packed_strided_integer_single(
                    x, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a);
            } else if (mtxfile->precision == mtx_double) {
                const struct mtxfile_vector_coordinate_integer_double * data =
                    mtxfile->data.vector_coordinate_integer_double;
                return mtxbasevector_init_packed_strided_integer_double(
                    x, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxfile->header.field == mtxfile_pattern) {
            const struct mtxfile_vector_coordinate_pattern * data =
                mtxfile->data.vector_coordinate_pattern;
            return mtxbasevector_init_packed_strided_pattern(
                x, size, num_nonzeros, sizeof(*data), 1, &data[0].i);
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxbasevector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxbasevector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    int err;
    if (mtxfmt == mtxfile_array) {
        if (x->field == mtx_field_real) {
            if (x->precision == mtx_single) {
                return mtxfile_init_vector_array_real_single(
                    mtxfile, x->size, x->data.real_single);
            } else if (x->precision == mtx_double) {
                return mtxfile_init_vector_array_real_double(
                    mtxfile, x->size, x->data.real_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_complex) {
            if (x->precision == mtx_single) {
                return mtxfile_init_vector_array_complex_single(
                    mtxfile, x->size, x->data.complex_single);
            } else if (x->precision == mtx_double) {
                return mtxfile_init_vector_array_complex_double(
                    mtxfile, x->size, x->data.complex_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_integer) {
            if (x->precision == mtx_single) {
                return mtxfile_init_vector_array_integer_single(
                    mtxfile, x->size, x->data.integer_single);
            } else if (x->precision == mtx_double) {
                return mtxfile_init_vector_array_integer_double(
                    mtxfile, x->size, x->data.integer_double);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_pattern) {
            return MTX_ERR_INCOMPATIBLE_FIELD;
        } else { return MTX_ERR_INVALID_FIELD; }
    } else if (mtxfmt == mtxfile_coordinate) {
        if (x->field == mtx_field_real) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_real, x->precision, x->size, x->num_nonzeros);
            if (err) return err;
            if (x->precision == mtx_single) {
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    mtxfile->data.vector_coordinate_real_single[k].i =
                        x->idx ? x->idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_real_single[k].a =
                        x->data.real_single[k];
                }
            } else if (x->precision == mtx_double) {
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    mtxfile->data.vector_coordinate_real_double[k].i =
                        x->idx ? x->idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_real_double[k].a =
                        x->data.real_double[k];
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_complex) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_complex, x->precision, x->size, x->num_nonzeros);
            if (err) return err;
            if (x->precision == mtx_single) {
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    mtxfile->data.vector_coordinate_complex_single[k].i =
                        x->idx ? x->idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_complex_single[k].a[0] =
                        x->data.complex_single[k][0];
                    mtxfile->data.vector_coordinate_complex_single[k].a[1] =
                        x->data.complex_single[k][1];
                }
            } else if (x->precision == mtx_double) {
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    mtxfile->data.vector_coordinate_complex_double[k].i =
                        x->idx ? x->idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_complex_double[k].a[0] =
                        x->data.complex_double[k][0];
                    mtxfile->data.vector_coordinate_complex_double[k].a[1] =
                        x->data.complex_double[k][1];
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_integer) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_integer, x->precision, x->size, x->num_nonzeros);
            if (err) return err;
            if (x->precision == mtx_single) {
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    mtxfile->data.vector_coordinate_integer_single[k].i =
                        x->idx ? x->idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_integer_single[k].a =
                        x->data.integer_single[k];
                }
            } else if (x->precision == mtx_double) {
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    mtxfile->data.vector_coordinate_integer_double[k].i =
                        x->idx ? x->idx[k]+1 : k+1;
                    mtxfile->data.vector_coordinate_integer_double[k].a =
                        x->data.integer_double[k];
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
        } else if (x->field == mtx_field_pattern) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, mtxfile_pattern, x->precision, x->size, x->num_nonzeros);
            if (err) return err;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                mtxfile->data.vector_coordinate_pattern[k].i =
                    x->idx ? x->idx[k]+1 : k+1;
        } else { return MTX_ERR_INVALID_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxbasevector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxbasevector’. If successful, then ‘dsts[p]’
 * points to a vector consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * Finally, the argument ‘invperm’ may either be ‘NULL’, in which case
 * it is ignored, or it must point to an array of length ‘size’, which
 * is used to store the inverse permutation obtained from sorting the
 * vector elements in ascending order according to their assigned
 * parts. That is, ‘invperm[i]’ is the original position (before
 * sorting) of the vector element that now occupies the ‘i’th position
 * among the sorted elements.
 *
 * The caller is responsible for calling ‘mtxbasevector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxbasevector_split(
    int num_parts,
    struct mtxbasevector ** dsts,
    const struct mtxbasevector * src,
    int64_t size,
    int * parts,
    int64_t * invperm)
{
    if (size != src->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    bool sorted = true;
    for (int64_t k = 0; k < size; k++) {
        if (parts[k] < 0 || parts[k] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        if (k > 0 && parts[k-1] > parts[k]) sorted = false;
    }

    /* sort by part number and invert the sorting permutation */
    bool free_invperm = !invperm;
    if (sorted) {
        if (!invperm) {
            invperm = malloc(size * sizeof(int64_t));
            if (!invperm) return MTX_ERR_ERRNO;
        }
        for (int64_t k = 0; k < size; k++) invperm[k] = k;
    } else {
        int64_t * perm = malloc(size * sizeof(int64_t));
        if (!perm) return MTX_ERR_ERRNO;
        int err = radix_sort_int(size, parts, perm);
        if (err) { free(perm); return err; }
        if (!invperm) {
            invperm = malloc(size * sizeof(int64_t));
            if (!invperm) { free(perm); return MTX_ERR_ERRNO; }
        }
        for (int64_t k = 0; k < size; k++) invperm[perm[k]] = k;
        free(perm);
    }

    /*
     * Extract each part by a) counting the number elements in the
     * part, b) allocating storage, and c) gathering vector elements
     * for the part.
     */
    int64_t offset = 0;
    for (int p = 0; p < num_parts; p++) {
        int64_t partsize = 0;
        while (offset+partsize < size && parts[offset+partsize] == p) partsize++;
        int err = mtxbasevector_alloc_packed(
            dsts[p], src->field, src->precision, src->size, partsize, &invperm[offset]);
        if (err) {
            for (int q = p-1; q >= 0; q--) mtxbasevector_free(dsts[q]);
            free(invperm);
            return err;
        }
        err = mtxbasevector_usga(dsts[p], src);
        if (err) {
            for (int q = p; q >= 0; q--) mtxbasevector_free(dsts[q]);
            if (free_invperm) free(invperm);
            return err;
        }
        if (src->idx) {
            for (int64_t k = 0; k < partsize; k++)
                dsts[p]->idx[k] = src->idx[invperm[offset+k]];
        }
        offset += partsize;
    }
    if (free_invperm) free(invperm);
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxbasevector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_swap(
    struct mtxbasevector * x,
    struct mtxbasevector * y)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((x->idx && !y->idx) || (!x->idx && y->idx)) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                float z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                double z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                float z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                double z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    if (x->idx && y->idx) {
        int64_t * xidx = x->idx;
        int64_t * yidx = y->idx;
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            int64_t z = yidx[k];
            yidx[k] = xidx[k];
            xidx[k] = z;
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_copy(
    struct mtxbasevector * y,
    const struct mtxbasevector * x)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (x->idx && !y->idx || !x->idx && y->idx) return MTX_ERR_INVALID_VECTOR_TYPE; */
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    if (x->idx && y->idx) {
        const int64_t * xidx = x->idx;
        int64_t * yidx = y->idx;
        for (int64_t k = 0; k < x->num_nonzeros; k++)
            yidx[k] = xidx[k];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxbasevector_sscal(
    float a,
    struct mtxbasevector * x,
    int64_t * num_flops)
{
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxbasevector_dscal(
    double a,
    struct mtxbasevector * x,
    int64_t * num_flops)
{
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxbasevector_cscal(
    float a[2],
    struct mtxbasevector * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxbasevector_zscal(
    double a[2],
    struct mtxbasevector * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        for (int64_t k = 0; k < x->num_nonzeros; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_saxpy(
    float a,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_daxpy(
    double a,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_caxpy(
    float a[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxbasevector_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_zaxpy(
    double a[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxbasevector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_saypx(
    float a,
    struct mtxbasevector * y,
    const struct mtxbasevector * x,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_daypx(
    double a,
    struct mtxbasevector * y,
    const struct mtxbasevector * x,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->num_nonzeros; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_sdot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_ddot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_cdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_cdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_zdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_zdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_cdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxbasevector_zdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_snrm2()’ computes the Euclidean norm of a vector
 * in single precision floating point.
 */
int mtxbasevector_snrm2(
    const struct mtxbasevector * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_dnrm2()’ computes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxbasevector_dnrm2(
    const struct mtxbasevector * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxbasevector_sasum(
    const struct mtxbasevector * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxbasevector_dasum(
    const struct mtxbasevector * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxbasevector_iamax(
    const struct mtxbasevector * x,
    int * iamax)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *iamax = 0;
            float max = x->num_nonzeros > 0 ? fabsf(xdata[0]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < fabsf(xdata[k])) {
                    max = fabsf(xdata[k]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *iamax = 0;
            double max = x->num_nonzeros > 0 ? fabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < fabs(xdata[k])) {
                    max = fabs(xdata[k]);
                    *iamax = k;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *iamax = 0;
            float max = x->num_nonzeros > 0 ? fabsf(xdata[0][0]) + fabsf(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < fabsf(xdata[k][0]) + fabsf(xdata[k][1])) {
                    max = fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *iamax = 0;
            double max = x->num_nonzeros > 0 ? fabs(xdata[0][0]) + fabs(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->num_nonzeros; k++) {
                if (max < fabs(xdata[k][0]) + fabs(xdata[k][1])) {
                    max = fabs(xdata[k][0]) + fabs(xdata[k][1]);
                    *iamax = k;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxbasevector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxbasevector_ussdot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float * dot,
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_sdot(x, y, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxbasevector_usddot(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double * dot,
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_ddot(x, y, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxbasevector_uscdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_cdotu(x, y, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_ussdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxbasevector_uszdotu(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_zdotu(x, y, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_usddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxbasevector_uscdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_cdotc(x, y, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_ussdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxbasevector_uszdotc(
    const struct mtxbasevector * x,
    const struct mtxbasevector * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_zdotc(x, y, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxbasevector_usddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxbasevector_ussaxpy(
    float alpha,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_saxpy(alpha, x, y, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxbasevector_usdaxpy(
    double alpha,
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_daxpy(alpha, x, y, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxbasevector_uscaxpy(
    float alpha[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_caxpy(alpha, x, y, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxbasevector_uszaxpy(
    double alpha[2],
    const struct mtxbasevector * x,
    struct mtxbasevector * y,
    int64_t * num_flops)
{
    if (!x->idx) return mtxbasevector_zaxpy(alpha, x, y, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_usga()’ performs a gather operation from a vector
 * ‘y’ in full storage format to a vector ‘x’ in packed form. Repeated
 * indices in the packed vector are allowed.
 */
int mtxbasevector_usga(
    struct mtxbasevector * x,
    const struct mtxbasevector * y)
{
    if (!x->idx) return mtxbasevector_copy(x, y);
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] = ydata[idx[k]];
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] = ydata[idx[k]];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] = ydata[idx[k]];
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                xdata[k] = ydata[idx[k]];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxbasevector_usgz(
    struct mtxbasevector * x,
    struct mtxbasevector * y)
{
    if (!x->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
                ydata[idx[k]][0] = 0;
                ydata[idx[k]][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
                ydata[idx[k]][0] = 0;
                ydata[idx[k]][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxbasevector_ussc(
    struct mtxbasevector * y,
    const struct mtxbasevector * x)
{
    if (!x->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    /* if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE; */
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] = xdata[k];
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] = xdata[k][0];
                ydata[idx[k]][1] = xdata[k][1];
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                ydata[idx[k]][0] = xdata[k][0];
                ydata[idx[k]][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] = xdata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                ydata[idx[k]] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxbasevector_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxbasevector_usscga(
    struct mtxbasevector * z,
    const struct mtxbasevector * x)
{
    if (!x->idx || !z->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != z->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != z->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != z->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    struct mtxbasevector y;
    int err = mtxbasevector_alloc(&y, x->field, x->precision, x->size);
    if (err) return err;
    err = mtxbasevector_setzero(&y);
    if (err) { mtxbasevector_free(&y); return err; }
    err = mtxbasevector_ussc(&y, x);
    if (err) { mtxbasevector_free(&y); return err; }
    err = mtxbasevector_usga(z, &y);
    if (err) { mtxbasevector_free(&y); return err; }
    mtxbasevector_free(&y);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxbasevector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxbasevector_recv()’.
 */
int mtxbasevector_send(
    const struct mtxbasevector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    if (offset + count > x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            err = MPI_Send(
                &x->data.real_single[offset], count, MPI_FLOAT,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Send(
                &x->data.real_double[offset], count, MPI_DOUBLE,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            err = MPI_Send(
                &x->data.complex_single[offset], 2*count, MPI_FLOAT,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Send(
                &x->data.complex_double[offset], 2*count, MPI_DOUBLE,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            err = MPI_Send(
                &x->data.integer_single[offset], count, MPI_INT32_T,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Send(
                &x->data.integer_double[offset], count, MPI_INT64_T,
                recipient, tag, comm);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxbasevector_send()’.
 */
int mtxbasevector_recv(
    struct mtxbasevector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode)
{
    int err;
    if (offset + count > x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            err = MPI_Recv(
                &x->data.real_single[offset], count, MPI_FLOAT,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Recv(
                &x->data.real_double[offset], count, MPI_DOUBLE,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            err = MPI_Recv(
                &x->data.complex_single[offset], 2*count, MPI_FLOAT,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Recv(
                &x->data.complex_double[offset], 2*count, MPI_DOUBLE,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            err = MPI_Recv(
                &x->data.integer_single[offset], count, MPI_INT32_T,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Recv(
                &x->data.integer_double[offset], count, MPI_INT64_T,
                sender, tag, comm, status);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxbasevector_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxbasevector_send()’.
 */
int mtxbasevector_irecv(
    struct mtxbasevector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode)
{
    int err;
    if (offset + count > x->num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            err = MPI_Irecv(
                &x->data.real_single[offset], count, MPI_FLOAT,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Irecv(
                &x->data.real_double[offset], count, MPI_DOUBLE,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            err = MPI_Irecv(
                &x->data.complex_single[offset], 2*count, MPI_FLOAT,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Irecv(
                &x->data.complex_double[offset], 2*count, MPI_DOUBLE,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            err = MPI_Irecv(
                &x->data.integer_single[offset], count, MPI_INT32_T,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else if (x->precision == mtx_double) {
            err = MPI_Irecv(
                &x->data.integer_double[offset], count, MPI_INT64_T,
                sender, tag, comm, request);
            if (mpierrcode) *mpierrcode = err;
            if (err) return MTX_ERR_MPI;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}
#endif
