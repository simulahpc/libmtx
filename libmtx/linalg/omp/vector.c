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
 * Data structures and routines for shared-memory parallel, dense
 * vectors using OpenMP.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_OPENMP
#include <libmtx/error.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/linalg/omp/vector.h>
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
 * ‘mtxompvector_field()’ gets the field of a vector.
 */
enum mtxfield mtxompvector_field(const struct mtxompvector * x)
{
    return mtxbasevector_field(&x->base);
}

/**
 * ‘mtxompvector_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxompvector_precision(const struct mtxompvector * x)
{
    return mtxbasevector_precision(&x->base);
}

/**
 * ‘mtxompvector_size()’ gets the size of a vector.
 */
int64_t mtxompvector_size(const struct mtxompvector * x)
{
    return mtxbasevector_size(&x->base);
}

/**
 * ‘mtxompvector_num_nonzeros()’ gets the number of explicitly stored
 * vector entries.
 */
int64_t mtxompvector_num_nonzeros(const struct mtxompvector * x)
{
    return mtxbasevector_num_nonzeros(&x->base);
}

/**
 * ‘mtxompvector_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxompvector_idx(const struct mtxompvector * x)
{
    return mtxbasevector_idx(&x->base);
}

/*
 * Memory management
 */

/**
 * ‘mtxompvector_free()’ frees storage allocated for a vector.
 */
void mtxompvector_free(
    struct mtxompvector * x)
{
    mtxbasevector_free(&x->base);
    free(x->offsets);
}

/**
 * ‘mtxompvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxompvector_alloc_copy(
    struct mtxompvector * dst,
    const struct mtxompvector * src)
{
    dst->num_threads = src->num_threads;
    if (src->offsets) {
        dst->offsets = malloc((dst->num_threads+1) * sizeof(int64_t));
        if (!dst->offsets) return MTX_ERR_ERRNO;
        for (int i = 0; i <= dst->num_threads; i++) {
            if (i > 0 && src->offsets[i] < src->offsets[i-1]) {
                free(dst->offsets);
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
            dst->offsets[i] = src->offsets[i];
        }
        if (dst->offsets[dst->num_threads] != src->base.num_nonzeros) {
            free(dst->offsets);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
        dst->sched = omp_sched_static;
        dst->chunk_size = 0;
    } else { dst->offsets = NULL; }
    dst->sched = src->sched;
    dst->chunk_size = src->chunk_size;
    int err = mtxbasevector_alloc_copy(&dst->base, &src->base);
    if (err) { free(dst->offsets); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxompvector_init_copy(
    struct mtxompvector * dst,
    const struct mtxompvector * src)
{
    dst->num_threads = src->num_threads;
    if (src->offsets) {
        dst->offsets = malloc((dst->num_threads+1) * sizeof(int64_t));
        if (!dst->offsets) return MTX_ERR_ERRNO;
        for (int i = 0; i <= dst->num_threads; i++) {
            if (i > 0 && src->offsets[i] < src->offsets[i-1]) {
                free(dst->offsets);
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
            dst->offsets[i] = src->offsets[i];
        }
        if (dst->offsets[dst->num_threads] != src->base.num_nonzeros) {
            free(dst->offsets);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
        dst->sched = omp_sched_static;
        dst->chunk_size = 0;
    } else { dst->offsets = NULL; }
    dst->sched = src->sched;
    dst->chunk_size = src->chunk_size;
    int err = mtxbasevector_alloc_copy(&dst->base, &src->base);
    if (err) { free(dst->offsets); return err; }
    return mtxbasevector_init_copy(&dst->base, &src->base);
}

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxompvector_alloc()’ allocates a vector.
 */
int mtxompvector_alloc(
    struct mtxompvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    x->num_threads = 0;
    x->offsets = NULL;
    x->sched = omp_sched_static;
    x->chunk_size = 0;
    return mtxbasevector_alloc(&x->base, field, precision, size);
}

/**
 * ‘mtxompvector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxompvector_init_real_single(
    struct mtxompvector * x,
    int64_t size,
    const float * data)
{
    int err = mtxompvector_alloc(
        x, mtx_field_real, mtx_single, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxompvector_init_real_double(
    struct mtxompvector * x,
    int64_t size,
    const double * data)
{
    int err = mtxompvector_alloc(
        x, mtx_field_real, mtx_double, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxompvector_init_complex_single(
    struct mtxompvector * x,
    int64_t size,
    const float (* data)[2])
{
    int err = mtxompvector_alloc(
        x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        base->data.complex_single[k][0] = data[k][0];
        base->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxompvector_init_complex_double(
    struct mtxompvector * x,
    int64_t size,
    const double (* data)[2])
{
    int err = mtxompvector_alloc(
        x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        base->data.complex_double[k][0] = data[k][0];
        base->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxompvector_init_integer_single(
    struct mtxompvector * x,
    int64_t size,
    const int32_t * data)
{
    int err = mtxompvector_alloc(
        x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxompvector_init_integer_double(
    struct mtxompvector * x,
    int64_t size,
    const int64_t * data)
{
    int err = mtxompvector_alloc(
        x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxompvector_init_pattern(
    struct mtxompvector * x,
    int64_t size)
{
    return mtxompvector_alloc(
        x, mtx_field_pattern, mtx_single, size);
}

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxompvector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxompvector_init_strided_real_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const float * data)
{
    int err = mtxompvector_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_single[k] = *(const float *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxompvector_init_strided_real_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const double * data)
{
    int err = mtxompvector_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_double[k] = *(const double *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxompvector_init_strided_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2])
{
    int err = mtxompvector_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        const void * p = ((const char *) data + k*stride);
        base->data.complex_single[k][0] = (*(const float (*)[2]) p)[0];
        base->data.complex_single[k][1] = (*(const float (*)[2]) p)[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxompvector_init_strided_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2])
{
    int err = mtxompvector_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        const void * p = ((const char *) data + k*stride);
        base->data.complex_double[k][0] = (*(const double (*)[2]) p)[0];
        base->data.complex_double[k][1] = (*(const double (*)[2]) p)[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxompvector_init_strided_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const int32_t * data)
{
    int err = mtxompvector_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_single[k] = *(const int32_t *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxompvector_init_strided_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t stride,
    const int64_t * data)
{
    int err = mtxompvector_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_double[k] = *(const int64_t *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/*
 * allocation and initialisation with custom schedule
 */

/**
 * ‘mtxompvector_alloc_custom()’ allocates a vector with a
 * user-defined schedule for parallel loops.
 *
 * If ‘offsets’ is ‘NULL’, then it is ignored. In this case, parallel
 * loops employ a user-defined schedule and chunk size, as specified
 * by ‘sched’ and ‘chunk_size’.
 *
 * Otherwise, a variable-sized block distribution of vector elements
 * is used. In this case, ‘offsets’ must point to an array of length
 * ‘num_threads+1’, containing the offsets to the first vector element
 * assigned to each thread. Moreover, ‘offsets[num_threads]’ must be
 * equal to the total number of vector elements.
 */
int mtxompvector_alloc_custom(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    if (offsets) {
        x->num_threads = num_threads;
        x->offsets = malloc((num_threads+1) * sizeof(int64_t));
        if (!x->offsets) return MTX_ERR_ERRNO;
        for (int i = 0; i <= num_threads; i++) {
            if (i > 0 && offsets[i] < offsets[i-1]) {
                free(x->offsets);
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            }
            x->offsets[i] = offsets[i];
        }
        if (x->offsets[num_threads] != size) {
            free(x->offsets);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
        x->sched = omp_sched_static;
        x->chunk_size = 0;
    } else {
        x->num_threads = 0;
        x->offsets = NULL;
        x->sched = sched;
        x->chunk_size = chunk_size;
    }
    int err = mtxbasevector_alloc(&x->base, field, precision, size);
    if (err) { free(x->offsets); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_custom_real_single()’ allocates and initialises
 * a vector with real, single precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_real_single(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float * data)
{
    int err = mtxompvector_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_real, mtx_single, size);
    if (err) return err;
    err = mtxompvector_set_real_single(x, size, stride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_custom_real_double()’ allocates and initialises
 * a vector with real, double precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_real_double(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double * data)
{
    int err = mtxompvector_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_real, mtx_double, size);
    if (err) return err;
    err = mtxompvector_set_real_double(x, size, stride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_custom_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_complex_single(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float (* data)[2])
{
    int err = mtxompvector_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_complex, mtx_single, size);
    if (err) return err;
    err = mtxompvector_set_complex_single(x, size, stride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_custom_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_complex_double(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double (* data)[2])
{
    int err = mtxompvector_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_complex, mtx_double, size);
    if (err) return err;
    err = mtxompvector_set_complex_double(x, size, stride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_custom_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_integer_single(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int32_t * data)
{
    int err = mtxompvector_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_integer, mtx_single, size);
    if (err) return err;
    err = mtxompvector_set_integer_single(x, size, stride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_custom_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 *
 * See also ‘mtxompvector_alloc_custom()’.
 */
int mtxompvector_init_custom_integer_double(
    struct mtxompvector * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int64_t * data)
{
    int err = mtxompvector_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_integer, mtx_double, size);
    if (err) return err;
    err = mtxompvector_set_integer_double(x, size, stride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}


/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxompvector_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxompvector_alloc_packed(
    struct mtxompvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    x->num_threads = 0;
    x->offsets = NULL;
    x->sched = omp_sched_static;
    x->chunk_size = 0;
    int err = mtxbasevector_alloc(&x->base, field, precision, size);
    if (err) return err;
    x->base.num_nonzeros = num_nonzeros;
    x->base.idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->base.idx) { mtxbasevector_free(&x->base); return MTX_ERR_ERRNO; }
    if (idx) {
        #pragma omp parallel for
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (idx[k] >= 0 && idx[k] < size) {
                x->base.idx[k] = idx[k];
            } else { mtxbasevector_free(&x->base); err = MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        if (err) return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxompvector_init_packed_real_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data)
{
    int err = mtxompvector_alloc_packed(
        x, mtx_field_real, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxompvector_init_packed_real_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data)
{
    int err = mtxompvector_alloc_packed(
        x, mtx_field_real, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxompvector_init_packed_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2])
{
    int err = mtxompvector_alloc_packed(
        x, mtx_field_complex, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++) {
        base->data.complex_single[k][0] = data[k][0];
        base->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxompvector_init_packed_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2])
{
    int err = mtxompvector_alloc_packed(
        x, mtx_field_complex, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++) {
        base->data.complex_double[k][0] = data[k][0];
        base->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxompvector_init_packed_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data)
{
    int err = mtxompvector_alloc_packed(
        x, mtx_field_integer, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxompvector_init_packed_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data)
{
    int err = mtxompvector_alloc_packed(
        x, mtx_field_integer, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxbasevector * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxompvector_init_packed_pattern(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    return mtxompvector_alloc_packed(
        x, mtx_field_pattern, mtx_single, size, num_nonzeros, idx);
}

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxompvector_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxompvector_alloc_packed_strided(
    struct mtxompvector * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    x->num_threads = 0;
    x->offsets = NULL;
    x->sched = omp_sched_static;
    x->chunk_size = 0;
    int err = mtxbasevector_alloc(&x->base, field, precision, size);
    if (err) return err;
    x->base.num_nonzeros = num_nonzeros;
    x->base.idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->base.idx) { mtxbasevector_free(&x->base); return MTX_ERR_ERRNO; }
    if (idx) {
        #pragma omp parallel for
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t idxk = (*(int64_t *) ((unsigned char *) idx + k*idxstride)) - idxbase;
            if (idxk >= 0 && idxk < size) {
                x->base.idx[k] = idxk;
            } else { mtxbasevector_free(&x->base); err = MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        if (err) return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxompvector_init_packed_strided_real_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data)
{
    int err = mtxompvector_alloc_packed_strided(
        x, mtx_field_real, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxompvector_set_real_single(x, num_nonzeros, datastride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxompvector_init_packed_strided_real_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data)
{
    int err = mtxompvector_alloc_packed_strided(
        x, mtx_field_real, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxompvector_set_real_double(x, num_nonzeros, datastride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxompvector_init_packed_strided_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2])
{
    int err = mtxompvector_alloc_packed_strided(
        x, mtx_field_complex, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxompvector_set_complex_single(x, num_nonzeros, datastride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxompvector_init_packed_strided_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2])
{
    int err = mtxompvector_alloc_packed_strided(
        x, mtx_field_complex, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxompvector_set_complex_double(x, num_nonzeros, datastride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxompvector_init_packed_strided_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data)
{
    int err = mtxompvector_alloc_packed_strided(
        x, mtx_field_integer, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxompvector_set_integer_single(x, num_nonzeros, datastride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxompvector_init_packed_strided_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data)
{
    int err = mtxompvector_alloc_packed_strided(
        x, mtx_field_integer, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxompvector_set_integer_double(x, num_nonzeros, datastride, data);
    if (err) { mtxompvector_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxompvector_init_packed_strided_pattern(
    struct mtxompvector * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    return mtxompvector_alloc_packed_strided(
        x, mtx_field_pattern, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
}

/*
 * accessing values
 */

/**
 * ‘mtxompvector_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_real_single(
    const struct mtxompvector * x,
    int64_t size,
    int stride,
    float * a)
{
    if (x->base.field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->base.num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    float * b = x->base.data.real_single;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                *(float *) ((char *) a + i*stride) = b[i];
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            *(float *) ((char *) a + i*stride) = b[i];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_real_double(
    const struct mtxompvector * x,
    int64_t size,
    int stride,
    double * a)
{
    if (x->base.field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->base.num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    double * b = x->base.data.real_double;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                *(double *) ((char *) a + i*stride) = b[i];
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            *(double *) ((char *) a + i*stride) = b[i];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    float (* a)[2])
{
    if (x->base.field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->base.num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    float (* b)[2] = x->base.data.complex_single;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++) {
                (*(float (*)[2])((char *) a + i*stride))[0] = b[i][0];
                (*(float (*)[2])((char *) a + i*stride))[1] = b[i][1];
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            (*(float (*)[2])((char *) a + i*stride))[0] = b[i][0];
            (*(float (*)[2])((char *) a + i*stride))[1] = b[i][1];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    double (* a)[2])
{
    if (x->base.field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->base.num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    double (* b)[2] = x->base.data.complex_double;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++) {
                (*(double (*)[2])((char *) a + i*stride))[0] = b[i][0];
                (*(double (*)[2])((char *) a + i*stride))[1] = b[i][1];
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            (*(double (*)[2])((char *) a + i*stride))[0] = b[i][0];
            (*(double (*)[2])((char *) a + i*stride))[1] = b[i][1];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    int32_t * a)
{
    if (x->base.field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (size < x->base.num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int32_t * b = x->base.data.integer_single;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                *(int32_t *)((char *) a + i*stride) = b[i];
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            *(int32_t *)((char *) a + i*stride) = b[i];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxompvector_get_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    int64_t * a)
{
    if (x->base.field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (size < x->base.num_nonzeros) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t * b = x->base.data.integer_double;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                *(int64_t *)((char *) a + i*stride) = b[i];
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            *(int64_t *)((char *) a + i*stride) = b[i];
    }
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxompvector_setzero()’ sets every value of a vector to zero.
 */
int mtxompvector_setzero(
    struct mtxompvector * xomp)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_single[i] = 0;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_single[i] = 0;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_double[i] = 0;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_double[i] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = 0;
                        x->data.complex_single[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = 0;
                    x->data.complex_single[i][1] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = 0;
                        x->data.complex_double[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = 0;
                    x->data.complex_double[i][1] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_single[i] = 0;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_single[i] = 0;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_double[i] = 0;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_double[i] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxompvector_set_constant_real_single(
    struct mtxompvector * xomp,
    float a)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = a;
                        x->data.complex_single[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = a;
                    x->data.complex_single[i][1] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = a;
                        x->data.complex_double[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = a;
                    x->data.complex_double[i][1] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxompvector_set_constant_real_double(
    struct mtxompvector * xomp,
    double a)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = a;
                        x->data.complex_single[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = a;
                    x->data.complex_single[i][1] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = a;
                        x->data.complex_double[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = a;
                    x->data.complex_double[i][1] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_constant_complex_single()’ sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtxompvector_set_constant_complex_single(
    struct mtxompvector * xomp,
    float a[2])
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = a[0];
                        x->data.complex_single[i][1] = a[1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = a[0];
                    x->data.complex_single[i][1] = a[1];
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = a[0];
                        x->data.complex_double[i][1] = a[1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = a[0];
                    x->data.complex_double[i][1] = a[1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_constant_complex_double()’ sets every value of a
 * vector equal to a constant, double precision floating point complex
 * number.
 */
int mtxompvector_set_constant_complex_double(
    struct mtxompvector * xomp,
    double a[2])
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = a[0];
                        x->data.complex_single[i][1] = a[1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = a[0];
                    x->data.complex_single[i][1] = a[1];
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = a[0];
                        x->data.complex_double[i][1] = a[1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = a[0];
                    x->data.complex_double[i][1] = a[1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_constant_integer_single()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxompvector_set_constant_integer_single(
    struct mtxompvector * xomp,
    int32_t a)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = a;
                        x->data.complex_single[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = a;
                    x->data.complex_single[i][1] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = a;
                        x->data.complex_double[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = a;
                    x->data.complex_double[i][1] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_constant_integer_double()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxompvector_set_constant_integer_double(
    struct mtxompvector * xomp,
    int64_t a)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.real_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.real_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_single[i][0] = a;
                        x->data.complex_single[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_single[i][0] = a;
                    x->data.complex_single[i][1] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++) {
                        x->data.complex_double[i][0] = a;
                        x->data.complex_double[i][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++) {
                    x->data.complex_double[i][0] = a;
                    x->data.complex_double[i][1] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_single[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_single[i] = a;
            }
        } else if (x->precision == mtx_double) {
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t i = xomp->offsets[t]; i < xomp->offsets[t+1]; i++)
                        x->data.integer_double[i] = a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t i = 0; i < x->size; i++)
                    x->data.integer_double[i] = a;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxompvector_set_real_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const float * a)
{
    if (x->base.field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->base.num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    float * b = x->base.data.real_single;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                b[i] = *(const float *) ((const char *) a + i*stride);
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            b[i] = *(const float *) ((const char *) a + i*stride);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxompvector_set_real_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const double * a)
{
    if (x->base.field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->base.num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    double * b = x->base.data.real_double;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                b[i] = *(const double *) ((const char *) a + i*stride);
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            b[i] = *(const double *) ((const char *) a + i*stride);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxompvector_set_complex_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    if (x->base.field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->base.num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    float (* b)[2] = x->base.data.complex_single;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++) {
                b[i][0] = (*(const float (*)[2])((const char *) a + i*stride))[0];
                b[i][1] = (*(const float (*)[2])((const char *) a + i*stride))[1];
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            b[i][0] = (*(const float (*)[2])((const char *) a + i*stride))[0];
            b[i][1] = (*(const float (*)[2])((const char *) a + i*stride))[1];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxompvector_set_complex_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    if (x->base.field != mtx_field_complex) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->base.num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    double (* b)[2] = x->base.data.complex_double;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++) {
                b[i][0] = (*(const double (*)[2])((const char *) a + i*stride))[0];
                b[i][1] = (*(const double (*)[2])((const char *) a + i*stride))[1];
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            b[i][0] = (*(const double (*)[2])((const char *) a + i*stride))[0];
            b[i][1] = (*(const double (*)[2])((const char *) a + i*stride))[1];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxompvector_set_integer_single(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const int32_t * a)
{
    if (x->base.field != mtx_field_real) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_single) return MTX_ERR_INVALID_PRECISION;
    if (x->base.num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int32_t * b = x->base.data.integer_single;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                b[i] = *(const int32_t *)((const char *) a + i*stride);
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            b[i] = *(const int32_t *)((const char *) a + i*stride);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxompvector_set_integer_double(
    struct mtxompvector * x,
    int64_t size,
    int stride,
    const int64_t * a)
{
    if (x->base.field != mtx_field_integer) return MTX_ERR_INVALID_FIELD;
    if (x->base.precision != mtx_double) return MTX_ERR_INVALID_PRECISION;
    if (x->base.num_nonzeros != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int64_t * b = x->base.data.integer_double;
    if (x->offsets) {
        #pragma omp parallel num_threads(x->num_threads)
        {
            int t = omp_get_thread_num();
            for (int64_t i = x->offsets[t]; i < x->offsets[t+1]; i++)
                b[i] = *(const int64_t *)((const char *) a + i*stride);
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            b[i] = *(const int64_t *)((const char *) a + i*stride);
    }
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxompvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxompvector_from_mtxfile(
    struct mtxompvector * x,
    const struct mtxfile * mtxfile)
{
    x->num_threads = 0;
    x->offsets = NULL;
    x->sched = omp_sched_static;
    x->chunk_size = 0;
    return mtxbasevector_from_mtxfile(&x->base, mtxfile);
}

/**
 * ‘mtxompvector_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxompvector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxompvector * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    return mtxbasevector_to_mtxfile(mtxfile, &x->base, num_rows, idx, mtxfmt);
}

/*
 * Partitioning
 */

/**
 * ‘mtxompvector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxompvector’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxompvector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxompvector_split(
    int num_parts,
    struct mtxompvector ** dsts,
    const struct mtxompvector * src,
    int64_t size,
    int * parts,
    int64_t * invperm)
{
    struct mtxbasevector ** basedsts = malloc(
        num_parts * sizeof(struct mtxbasevector *));
    if (!basedsts) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        dsts[p]->num_threads = src->num_threads;
        dsts[p]->offsets = NULL;
        dsts[p]->chunk_size = 0;
        basedsts[p] = &dsts[p]->base;
    }
    int err = mtxbasevector_split(
        num_parts, basedsts, &src->base, size, parts, invperm);
    free(basedsts);
    return err;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxompvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_swap(
    struct mtxompvector * xomp,
    struct mtxompvector * yomp)
{
    struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((x->idx && !y->idx) || (!x->idx && y->idx)) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        float z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    float z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                }
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        double z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    double z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        float z0 = ydata[k][0]; ydata[k][0] = xdata[k][0]; xdata[k][0] = z0;
                        float z1 = ydata[k][1]; ydata[k][1] = xdata[k][1]; xdata[k][1] = z1;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    float z0 = ydata[k][0]; ydata[k][0] = xdata[k][0]; xdata[k][0] = z0;
                    float z1 = ydata[k][1]; ydata[k][1] = xdata[k][1]; xdata[k][1] = z1;
                }
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        double z0 = ydata[k][0]; ydata[k][0] = xdata[k][0]; xdata[k][0] = z0;
                        double z1 = ydata[k][1]; ydata[k][1] = xdata[k][1]; xdata[k][1] = z1;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    double z0 = ydata[k][0]; ydata[k][0] = xdata[k][0]; xdata[k][0] = z0;
                    double z1 = ydata[k][1]; ydata[k][1] = xdata[k][1]; xdata[k][1] = z1;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        int32_t z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    int32_t z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                }
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        int64_t z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    int64_t z = ydata[k]; ydata[k] = xdata[k]; xdata[k] = z;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    if (x->idx && y->idx) {
        int64_t * xidx = x->idx;
        int64_t * yidx = y->idx;
        if (yomp->offsets) {
            #pragma omp parallel num_threads(yomp->num_threads)
            {
                int t = omp_get_thread_num();
                for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                    int64_t z = yidx[k]; yidx[k] = xidx[k]; xidx[k] = z;
                }
            }
        } else {
            #pragma omp parallel for
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                int64_t z = yidx[k]; yidx[k] = xidx[k]; xidx[k] = z;
            }
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_copy(
    struct mtxompvector * yomp,
    const struct mtxompvector * xomp)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if ((x->idx && !y->idx) || (!x->idx && y->idx)) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[k] = xdata[k];
            }
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[k] = xdata[k];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] = xdata[k][0]; ydata[k][1] = xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[k][0] = xdata[k][0]; ydata[k][1] = xdata[k][1];
                }
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] = xdata[k][0]; ydata[k][1] = xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[k][0] = xdata[k][0]; ydata[k][1] = xdata[k][1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[k] = xdata[k];
            }
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[k] = xdata[k];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    if (x->idx && y->idx) {
        const int64_t * xidx = x->idx;
        int64_t * yidx = y->idx;
        if (yomp->offsets) {
            #pragma omp parallel num_threads(yomp->num_threads)
            {
                int t = omp_get_thread_num();
                for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        yidx[k] = xidx[k];
            }
        } else {
            #pragma omp parallel for
            for (int64_t k = 0; k < x->num_nonzeros; k++)
                yidx[k] = xidx[k];
        }
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxompvector_sscal(
    float a,
    struct mtxompvector * xomp,
    int64_t * num_flops)
{
    struct mtxbasevector * x = &xomp->base;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] *= a; xdata[k][1] *= a;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] *= a; xdata[k][1] *= a;
                }
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] *= a; xdata[k][1] *= a;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] *= a; xdata[k][1] *= a;
                }
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxompvector_dscal(
    double a,
    struct mtxompvector * xomp,
    int64_t * num_flops)
{
    struct mtxbasevector * x = &xomp->base;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] *= a; xdata[k][1] *= a;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] *= a; xdata[k][1] *= a;
                }
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] *= a; xdata[k][1] *= a;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] *= a; xdata[k][1] *= a;
                }
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] *= a;
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] *= a;
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxompvector_cscal(
    float a[2],
    struct mtxompvector * xomp,
    int64_t * num_flops)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        if (xomp->offsets) {
            #pragma omp parallel num_threads(xomp->num_threads)
            {
                int t = omp_get_thread_num();
                for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                    float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                    float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                    xdata[k][0] = c;
                    xdata[k][1] = d;
                }
            }
        } else {
            #pragma omp parallel for
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                xdata[k][0] = c;
                xdata[k][1] = d;
            }
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        if (xomp->offsets) {
            #pragma omp parallel num_threads(xomp->num_threads)
            {
                int t = omp_get_thread_num();
                for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                    double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                    double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                    xdata[k][0] = c;
                    xdata[k][1] = d;
                }
            }
        } else {
            #pragma omp parallel for
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                xdata[k][0] = c;
                xdata[k][1] = d;
            }
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxompvector_zscal(
    double a[2],
    struct mtxompvector * xomp,
    int64_t * num_flops)
{
    struct mtxbasevector * x = &xomp->base;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        if (xomp->offsets) {
            #pragma omp parallel num_threads(xomp->num_threads)
            {
                int t = omp_get_thread_num();
                for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                    float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                    float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                    xdata[k][0] = c;
                    xdata[k][1] = d;
                }
            }
        } else {
            #pragma omp parallel for
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                xdata[k][0] = c;
                xdata[k][1] = d;
            }
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        if (xomp->offsets) {
            #pragma omp parallel num_threads(xomp->num_threads)
            {
                int t = omp_get_thread_num();
                for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                    double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                    double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                    xdata[k][0] = c;
                    xdata[k][1] = d;
                }
            }
        } else {
            #pragma omp parallel for
            for (int64_t k = 0; k < x->num_nonzeros; k++) {
                double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
                double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
                xdata[k][0] = c;
                xdata[k][1] = d;
            }
        }
        if (num_flops) *num_flops += 6*x->num_nonzeros;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_saxpy(
    float a,
    const struct mtxompvector * xomp,
    struct mtxompvector * yomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] += a*xdata[k][0];
                        ydata[k][1] += a*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] += a*xdata[k][0];
                    ydata[k][1] += a*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] += a*xdata[k][0];
                        ydata[k][1] += a*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] += a*xdata[k][0];
                    ydata[k][1] += a*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_daxpy(
    double a,
    const struct mtxompvector * xomp,
    struct mtxompvector * yomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] += a*xdata[k][0];
                        ydata[k][1] += a*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] += a*xdata[k][0];
                    ydata[k][1] += a*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] += a*xdata[k][0];
                        ydata[k][1] += a*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] += a*xdata[k][0];
                    ydata[k][1] += a*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] += a*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] += a*xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_caxpy(
    float a[2],
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxompvector_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_zaxpy(
    double a[2],
    const struct mtxompvector * x,
    struct mtxompvector * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxompvector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_saypx(
    float a,
    struct mtxompvector * yomp,
    const struct mtxompvector * xomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                        ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                    ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                        ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                    ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_daypx(
    double a,
    struct mtxompvector * yomp,
    const struct mtxompvector * xomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                        ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                    ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++) {
                        ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                        ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++) {
                    ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                    ydata[k][1] = a*ydata[k][1]+xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (yomp->offsets) {
                #pragma omp parallel num_threads(yomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = yomp->offsets[t]; k < yomp->offsets[t+1]; k++)
                        ydata[k] = a*ydata[k]+xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < y->num_nonzeros; k++)
                    ydata[k] = a*ydata[k]+xdata[k];
            }
            if (num_flops) *num_flops += 2*y->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_sdot(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    float * dot,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    float c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *dot = c;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_ddot(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    double * dot,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    double c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *dot = c;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_cdotu(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    float c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxompvector_sdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_zdotu(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    double c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxompvector_ddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_cdotc(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    float c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxompvector_sdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxompvector_zdotc(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    double c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                        c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                    c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxompvector_ddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxompvector_snrm2(
    const struct mtxompvector * xomp,
    float * nrm2,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    float c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *nrm2 = sqrtf(c);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxompvector_dnrm2(
    const struct mtxompvector * xomp,
    double * nrm2,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    double c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*xdata[k];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *nrm2 = sqrt(c);
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxompvector_sasum(
    const struct mtxompvector * xomp,
    float * asum,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    float c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabsf(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabsf(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabs(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabs(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += abs(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += abs(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += llabs(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += llabs(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *asum = c;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxompvector_dasum(
    const struct mtxompvector * xomp,
    double * asum,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    double c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabsf(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabsf(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabs(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabs(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += abs(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += abs(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += llabs(xdata[k]);
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += llabs(xdata[k]);
            }
            if (num_flops) *num_flops += x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *asum = c;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the vector is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxompvector_iamax(
    const struct mtxompvector * xomp,
    int * iamax)
{
    const struct mtxbasevector * x = &xomp->base;
    return mtxbasevector_iamax(x, iamax);
}

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxompvector_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_ussdot(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    float * dot,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_sdot(xomp, yomp, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    float c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *dot = c;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_usddot(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    double * dot,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_ddot(xomp, yomp, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    double c = 0;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        c += xdata[k]*ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for reduction(+:c)
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    c += xdata[k]*ydata[idx[k]];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    *dot = c;
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uscdotu(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_cdotu(xomp, yomp, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    float c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
        (*dot)[0] = c0; (*dot)[1] = c1;
    } else {
        (*dot)[1] = 0;
        return mtxompvector_ussdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uszdotu(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_zdotu(xomp, yomp, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    double c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
        (*dot)[0] = c0; (*dot)[1] = c1;
    } else {
        (*dot)[1] = 0;
        return mtxompvector_usddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uscdotc(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_cdotc(xomp, yomp, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    float c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
        (*dot)[0] = c0; (*dot)[1] = c1;
    } else {
        (*dot)[1] = 0;
        return mtxompvector_ussdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxompvector_uszdotc(
    const struct mtxompvector * xomp,
    const struct mtxompvector * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_zdotc(xomp, yomp, dot, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    double c0 = 0, c1 = 0;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads) reduction(+:c0,c1)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                        c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:c0,c1)
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                    c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
        (*dot)[0] = c0; (*dot)[1] = c1;
    } else {
        (*dot)[1] = 0;
        return mtxompvector_usddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_ussaxpy(
    float alpha,
    const struct mtxompvector * xomp,
    struct mtxompvector * yomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_saxpy(alpha, xomp, yomp, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha*xdata[k][0];
                        ydata[idx[k]][1] += alpha*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha*xdata[k][0];
                    ydata[idx[k]][1] += alpha*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha*xdata[k][0];
                        ydata[idx[k]][1] += alpha*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha*xdata[k][0];
                    ydata[idx[k]][1] += alpha*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_usdaxpy(
    double alpha,
    const struct mtxompvector * xomp,
    struct mtxompvector * yomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_daxpy(alpha, xomp, yomp, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha*xdata[k][0];
                        ydata[idx[k]][1] += alpha*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha*xdata[k][0];
                    ydata[idx[k]][1] += alpha*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha*xdata[k][0];
                        ydata[idx[k]][1] += alpha*xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha*xdata[k][0];
                    ydata[idx[k]][1] += alpha*xdata[k][1];
                }
            }
            if (num_flops) *num_flops += 4*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] += alpha*xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] += alpha*xdata[k];
            }
            if (num_flops) *num_flops += 2*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_uscaxpy(
    float alpha[2],
    const struct mtxompvector * xomp,
    struct mtxompvector * yomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_caxpy(alpha, xomp, yomp, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                        ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                    ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                        ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                    ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxompvector_uszaxpy(
    double alpha[2],
    const struct mtxompvector * xomp,
    struct mtxompvector * yomp,
    int64_t * num_flops)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxompvector_zaxpy(alpha, xomp, yomp, num_flops);
    if (y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                        ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                    ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                        ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                    ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
                }
            }
            if (num_flops) *num_flops += 8*x->num_nonzeros;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxompvector_usga(
    struct mtxompvector * xomp,
    const struct mtxompvector * yomp)
{
    struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx) return mtxbasevector_copy(x, y);
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] = ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] = ydata[idx[k]];
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] = ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] = ydata[idx[k]];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] = ydata[idx[k]][0];
                        xdata[k][1] = ydata[idx[k]][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] = ydata[idx[k]][0];
                    xdata[k][1] = ydata[idx[k]][1];
                }
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] = ydata[idx[k]][0];
                        xdata[k][1] = ydata[idx[k]][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] = ydata[idx[k]][0];
                    xdata[k][1] = ydata[idx[k]][1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] = ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] = ydata[idx[k]];
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        xdata[k] = ydata[idx[k]];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    xdata[k] = ydata[idx[k]];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxompvector_usgz(
    struct mtxompvector * xomp,
    struct mtxompvector * yomp)
{
    struct mtxbasevector * x = &xomp->base;
    const struct mtxbasevector * y = &yomp->base;
    if (!x->idx || y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] = ydata[idx[k]][0]; ydata[idx[k]][0] = 0;
                        xdata[k][1] = ydata[idx[k]][1]; ydata[idx[k]][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] = ydata[idx[k]][0]; ydata[idx[k]][0] = 0;
                    xdata[k][1] = ydata[idx[k]][1]; ydata[idx[k]][1] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k][0] = ydata[idx[k]][0]; ydata[idx[k]][0] = 0;
                        xdata[k][1] = ydata[idx[k]][1]; ydata[idx[k]][1] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k][0] = ydata[idx[k]][0]; ydata[idx[k]][0] = 0;
                    xdata[k][1] = ydata[idx[k]][1]; ydata[idx[k]][1] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                }
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    xdata[k] = ydata[idx[k]]; ydata[idx[k]] = 0;
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxompvector_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxompvector_ussc(
    struct mtxompvector * yomp,
    const struct mtxompvector * xomp)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * y = &yomp->base;
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
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]] = xdata[k];
                }
            }
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] = xdata[k];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] = xdata[k][0];
                        ydata[idx[k]][1] = xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] = xdata[k][0];
                    ydata[idx[k]][1] = xdata[k][1];
                }
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++) {
                        ydata[idx[k]][0] = xdata[k][0];
                        ydata[idx[k]][1] = xdata[k][1];
                    }
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++) {
                    ydata[idx[k]][0] = xdata[k][0];
                    ydata[idx[k]][1] = xdata[k][1];
                }
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] = xdata[k];
            }
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            if (xomp->offsets) {
                #pragma omp parallel num_threads(xomp->num_threads)
                {
                    int t = omp_get_thread_num();
                    for (int64_t k = xomp->offsets[t]; k < xomp->offsets[t+1]; k++)
                        ydata[idx[k]] = xdata[k];
                }
            } else {
                #pragma omp parallel for
                for (int64_t k = 0; k < x->num_nonzeros; k++)
                    ydata[idx[k]] = xdata[k];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxompvector_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxompvector_usscga(
    struct mtxompvector * zomp,
    const struct mtxompvector * xomp)
{
    const struct mtxbasevector * x = &xomp->base;
    struct mtxbasevector * z = &zomp->base;
    if (!x->idx || !z->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != z->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != z->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != z->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    struct mtxompvector y;
    int err = mtxompvector_alloc(&y, x->field, x->precision, x->size);
    if (err) return err;
    err = mtxompvector_setzero(&y);
    if (err) { mtxompvector_free(&y); return err; }
    err = mtxompvector_ussc(&y, xomp);
    if (err) { mtxompvector_free(&y); return err; }
    err = mtxompvector_usga(zomp, &y);
    if (err) { mtxompvector_free(&y); return err; }
    mtxompvector_free(&y);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxompvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxompvector_recv()’.
 */
int mtxompvector_send(
    const struct mtxompvector * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    return mtxbasevector_send(
        &x->base, offset, count, recipient, tag, comm, mpierrcode);
}

/**
 * ‘mtxompvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxompvector_send()’.
 */
int mtxompvector_recv(
    struct mtxompvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode)
{
    return mtxbasevector_recv(
        &x->base, offset, count, sender, tag, comm, status, mpierrcode);
}

/**
 * ‘mtxompvector_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxompvector_send()’.
 */
int mtxompvector_irecv(
    struct mtxompvector * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode)
{
    return mtxbasevector_irecv(
        &x->base, offset, count, sender, tag, comm, request, mpierrcode);
}
#endif
#endif
