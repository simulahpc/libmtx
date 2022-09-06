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
 * Last modified: 2022-07-12
 *
 * Data structures and routines for shared-memory parallel, dense
 * vectors using OpenMP.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_OPENMP
#include <libmtx/error.h>
#include <libmtx/vector/precision.h>
#include <libmtx/vector/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/omp.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/vector.h>

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
 * ‘mtxvector_omp_field()’ gets the field of a vector.
 */
enum mtxfield mtxvector_omp_field(const struct mtxvector_omp * x)
{
    return mtxvector_base_field(&x->base);
}

/**
 * ‘mtxvector_omp_precision()’ gets the precision of a vector.
 */
enum mtxprecision mtxvector_omp_precision(const struct mtxvector_omp * x)
{
    return mtxvector_base_precision(&x->base);
}

/**
 * ‘mtxvector_omp_size()’ gets the size of a vector.
 */
int64_t mtxvector_omp_size(const struct mtxvector_omp * x)
{
    return mtxvector_base_size(&x->base);
}

/**
 * ‘mtxvector_omp_num_nonzeros()’ gets the number of explicitly stored
 * vector entries.
 */
int64_t mtxvector_omp_num_nonzeros(const struct mtxvector_omp * x)
{
    return mtxvector_base_num_nonzeros(&x->base);
}

/**
 * ‘mtxvector_omp_idx()’ gets a pointer to an array containing the
 * offset of each nonzero vector entry for a vector in packed storage
 * format.
 */
int64_t * mtxvector_omp_idx(const struct mtxvector_omp * x)
{
    return mtxvector_base_idx(&x->base);
}

/*
 * Memory management
 */

/**
 * ‘mtxvector_omp_free()’ frees storage allocated for a vector.
 */
void mtxvector_omp_free(
    struct mtxvector_omp * x)
{
    mtxvector_base_free(&x->base);
    free(x->offsets);
}

/**
 * ‘mtxvector_omp_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_omp_alloc_copy(
    struct mtxvector_omp * dst,
    const struct mtxvector_omp * src)
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
    int err = mtxvector_base_alloc_copy(&dst->base, &src->base);
    if (err) { free(dst->offsets); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_omp_init_copy(
    struct mtxvector_omp * dst,
    const struct mtxvector_omp * src)
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
    int err = mtxvector_base_alloc_copy(&dst->base, &src->base);
    if (err) { free(dst->offsets); return err; }
    return mtxvector_base_init_copy(&dst->base, &src->base);
}

/*
 * initialise vectors in full storage format
 */

/**
 * ‘mtxvector_omp_alloc()’ allocates a vector.
 */
int mtxvector_omp_alloc(
    struct mtxvector_omp * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    x->num_threads = 0;
    x->offsets = NULL;
    x->sched = omp_sched_static;
    x->chunk_size = 0;
    return mtxvector_base_alloc(&x->base, field, precision, size);
}

/**
 * ‘mtxvector_omp_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_omp_init_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float * data)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_real, mtx_single, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_omp_init_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    const double * data)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_real, mtx_double, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float (* data)[2])
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        base->data.complex_single[k][0] = data[k][0];
        base->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    const double (* data)[2])
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        base->data.complex_double[k][0] = data[k][0];
        base->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    const int32_t * data)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    const int64_t * data)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_omp_init_pattern(
    struct mtxvector_omp * x,
    int64_t size)
{
    return mtxvector_omp_alloc(
        x, mtx_field_pattern, mtx_single, size);
}

/*
 * initialise vectors in full storage format from strided arrays
 */

/**
 * ‘mtxvector_omp_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_omp_init_strided_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float * data)
{
    int err = mtxvector_omp_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_single[k] = *(const float *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_omp_init_strided_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const double * data)
{
    int err = mtxvector_omp_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.real_double[k] = *(const double *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_strided_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2])
{
    int err = mtxvector_omp_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        const void * p = ((const char *) data + k*stride);
        base->data.complex_single[k][0] = (*(const float (*)[2]) p)[0];
        base->data.complex_single[k][1] = (*(const float (*)[2]) p)[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_strided_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2])
{
    int err = mtxvector_omp_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++) {
        const void * p = ((const char *) data + k*stride);
        base->data.complex_double[k][0] = (*(const double (*)[2]) p)[0];
        base->data.complex_double[k][1] = (*(const double (*)[2]) p)[1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_strided_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const int32_t * data)
{
    int err = mtxvector_omp_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_single[k] = *(const int32_t *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_strided_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const int64_t * data)
{
    int err = mtxvector_omp_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < size; k++)
        base->data.integer_double[k] = *(const int64_t *) ((const char *) data + k*stride);
    return MTX_SUCCESS;
}

/*
 * allocation and initialisation with custom schedule
 */

/**
 * ‘mtxvector_omp_alloc_custom()’ allocates a vector with a
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
int mtxvector_omp_alloc_custom(
    struct mtxvector_omp * x,
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
    int err = mtxvector_base_alloc(&x->base, field, precision, size);
    if (err) { free(x->offsets); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_custom_real_single()’ allocates and initialises
 * a vector with real, single precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_real_single(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float * data)
{
    int err = mtxvector_omp_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_real, mtx_single, size);
    if (err) return err;
    err = mtxvector_omp_set_real_single(x, size, stride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_custom_real_double()’ allocates and initialises
 * a vector with real, double precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_real_double(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double * data)
{
    int err = mtxvector_omp_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_real, mtx_double, size);
    if (err) return err;
    err = mtxvector_omp_set_real_double(x, size, stride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_custom_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_complex_single(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const float (* data)[2])
{
    int err = mtxvector_omp_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_complex, mtx_single, size);
    if (err) return err;
    err = mtxvector_omp_set_complex_single(x, size, stride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_custom_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_complex_double(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const double (* data)[2])
{
    int err = mtxvector_omp_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_complex, mtx_double, size);
    if (err) return err;
    err = mtxvector_omp_set_complex_double(x, size, stride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_custom_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_integer_single(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int32_t * data)
{
    int err = mtxvector_omp_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_integer, mtx_single, size);
    if (err) return err;
    err = mtxvector_omp_set_integer_single(x, size, stride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_custom_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 *
 * See also ‘mtxvector_omp_alloc_custom()’.
 */
int mtxvector_omp_init_custom_integer_double(
    struct mtxvector_omp * x,
    int num_threads,
    const int64_t * offsets,
    omp_sched_t sched,
    int chunk_size,
    int64_t size,
    int stride,
    const int64_t * data)
{
    int err = mtxvector_omp_alloc_custom(
        x, num_threads, offsets, sched, chunk_size, mtx_field_integer, mtx_double, size);
    if (err) return err;
    err = mtxvector_omp_set_integer_double(x, size, stride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}


/*
 * initialise vectors in packed storage format
 */

/**
 * ‘mtxvector_omp_alloc_packed()’ allocates a vector in packed
 * storage format.
 */
int mtxvector_omp_alloc_packed(
    struct mtxvector_omp * x,
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
    int err = mtxvector_base_alloc(&x->base, field, precision, size);
    if (err) return err;
    x->base.num_nonzeros = num_nonzeros;
    x->base.idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->base.idx) { mtxvector_base_free(&x->base); return MTX_ERR_ERRNO; }
    if (idx) {
        #pragma omp parallel for
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (idx[k] >= 0 && idx[k] < size) {
                x->base.idx[k] = idx[k];
            } else { mtxvector_base_free(&x->base); err = MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        if (err) return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_omp_init_packed_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data)
{
    int err = mtxvector_omp_alloc_packed(
        x, mtx_field_real, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_omp_init_packed_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data)
{
    int err = mtxvector_omp_alloc_packed(
        x, mtx_field_real, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_packed_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2])
{
    int err = mtxvector_omp_alloc_packed(
        x, mtx_field_complex, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++) {
        base->data.complex_single[k][0] = data[k][0];
        base->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_packed_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2])
{
    int err = mtxvector_omp_alloc_packed(
        x, mtx_field_complex, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++) {
        base->data.complex_double[k][0] = data[k][0];
        base->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_packed_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data)
{
    int err = mtxvector_omp_alloc_packed(
        x, mtx_field_integer, mtx_single, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_packed_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data)
{
    int err = mtxvector_omp_alloc_packed(
        x, mtx_field_integer, mtx_double, size, num_nonzeros, idx);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for
    for (int64_t k = 0; k < num_nonzeros; k++)
        base->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxvector_omp_init_packed_pattern(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    return mtxvector_omp_alloc_packed(
        x, mtx_field_pattern, mtx_single, size, num_nonzeros, idx);
}

/*
 * initialise vectors in packed storage format from strided arrays
 */

/**
 * ‘mtxvector_omp_alloc_packed_strided()’ allocates a vector in
 * packed storage format.
 */
int mtxvector_omp_alloc_packed_strided(
    struct mtxvector_omp * x,
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
    int err = mtxvector_base_alloc(&x->base, field, precision, size);
    if (err) return err;
    x->base.num_nonzeros = num_nonzeros;
    x->base.idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->base.idx) { mtxvector_base_free(&x->base); return MTX_ERR_ERRNO; }
    if (idx) {
        #pragma omp parallel for
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t idxk = (*(int64_t *) ((unsigned char *) idx + k*idxstride)) - idxbase;
            if (idxk >= 0 && idxk < size) {
                x->base.idx[k] = idxk;
            } else { mtxvector_base_free(&x->base); err = MTX_ERR_INDEX_OUT_OF_BOUNDS; }
        }
        if (err) return err;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_omp_init_packed_strided_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float * data)
{
    int err = mtxvector_omp_alloc_packed_strided(
        x, mtx_field_real, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxvector_omp_set_real_single(x, num_nonzeros, datastride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_omp_init_packed_strided_real_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double * data)
{
    int err = mtxvector_omp_alloc_packed_strided(
        x, mtx_field_real, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxvector_omp_set_real_double(x, num_nonzeros, datastride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_omp_init_packed_strided_complex_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const float (* data)[2])
{
    int err = mtxvector_omp_alloc_packed_strided(
        x, mtx_field_complex, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxvector_omp_set_complex_single(x, num_nonzeros, datastride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_omp_init_packed_strided_complex_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const double (* data)[2])
{
    int err = mtxvector_omp_alloc_packed_strided(
        x, mtx_field_complex, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxvector_omp_set_complex_double(x, num_nonzeros, datastride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_omp_init_packed_strided_integer_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int32_t * data)
{
    int err = mtxvector_omp_alloc_packed_strided(
        x, mtx_field_integer, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxvector_omp_set_integer_single(x, num_nonzeros, datastride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_omp_init_packed_strided_integer_double(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx,
    int datastride,
    const int64_t * data)
{
    int err = mtxvector_omp_alloc_packed_strided(
        x, mtx_field_integer, mtx_double, size, num_nonzeros, idxstride, idxbase, idx);
    if (err) return err;
    err = mtxvector_omp_set_integer_double(x, num_nonzeros, datastride, data);
    if (err) { mtxvector_omp_free(x); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_packed_pattern()’ allocates and initialises a
 * binary pattern vector, where every nonzero entry has a value of
 * one.
 */
int mtxvector_omp_init_packed_strided_pattern(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * idx)
{
    return mtxvector_omp_alloc_packed_strided(
        x, mtx_field_pattern, mtx_single, size, num_nonzeros, idxstride, idxbase, idx);
}

/*
 * accessing values
 */

/**
 * ‘mtxvector_omp_get_real_single()’ obtains the values of a vector
 * of single precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_real_single(
    const struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_get_real_double()’ obtains the values of a vector
 * of double precision floating point numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_real_double(
    const struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_get_complex_single()’ obtains the values of a
 * vector of single precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_complex_single(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_get_complex_double()’ obtains the values of a
 * vector of double precision floating point complex numbers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_complex_double(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_get_integer_single()’ obtains the values of a
 * vector of single precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_integer_single(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_get_integer_double()’ obtains the values of a
 * vector of double precision integers.
 *
 * The array ‘a’ must be large enough to store ‘size’ elements
 * separated by the given stride (in bytes), and ‘size’ must be
 * greater than or equal to the number of elements in the vector.
 */
int mtxvector_omp_get_integer_double(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_setzero()’ sets every value of a vector to zero.
 */
int mtxvector_omp_setzero(
    struct mtxvector_omp * xomp)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_omp_set_constant_real_single(
    struct mtxvector_omp * xomp,
    float a)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_omp_set_constant_real_double(
    struct mtxvector_omp * xomp,
    double a)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_constant_complex_single()’ sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtxvector_omp_set_constant_complex_single(
    struct mtxvector_omp * xomp,
    float a[2])
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_constant_complex_double()’ sets every value of a
 * vector equal to a constant, double precision floating point complex
 * number.
 */
int mtxvector_omp_set_constant_complex_double(
    struct mtxvector_omp * xomp,
    double a[2])
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_constant_integer_single()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxvector_omp_set_constant_integer_single(
    struct mtxvector_omp * xomp,
    int32_t a)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_constant_integer_double()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxvector_omp_set_constant_integer_double(
    struct mtxvector_omp * xomp,
    int64_t a)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxvector_omp_set_real_single(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxvector_omp_set_real_double(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxvector_omp_set_complex_single(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxvector_omp_set_complex_double(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_omp_set_integer_single(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_omp_set_integer_double(
    struct mtxvector_omp * x,
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
 * ‘mtxvector_omp_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_omp_from_mtxfile(
    struct mtxvector_omp * x,
    const struct mtxfile * mtxfile)
{
    x->num_threads = 0;
    x->offsets = NULL;
    x->sched = omp_sched_static;
    x->chunk_size = 0;
    return mtxvector_base_from_mtxfile(&x->base, mtxfile);
}

/**
 * ‘mtxvector_omp_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_omp_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_omp * x,
    int64_t num_rows,
    const int64_t * idx,
    enum mtxfileformat mtxfmt)
{
    return mtxvector_base_to_mtxfile(mtxfile, &x->base, num_rows, idx, mtxfmt);
}

/*
 * Partitioning
 */

/**
 * ‘mtxvector_omp_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxvector_omp’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxvector_omp_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxvector_omp_split(
    int num_parts,
    struct mtxvector_omp ** dsts,
    const struct mtxvector_omp * src,
    int64_t size,
    int * parts,
    int64_t * invperm)
{
    struct mtxvector_base ** basedsts = malloc(
        num_parts * sizeof(struct mtxvector_base *));
    if (!basedsts) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        dsts[p]->num_threads = src->num_threads;
        dsts[p]->offsets = NULL;
        dsts[p]->chunk_size = 0;
        basedsts[p] = &dsts[p]->base;
    }
    int err = mtxvector_base_split(
        num_parts, basedsts, &src->base, size, parts, invperm);
    free(basedsts);
    return err;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_omp_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_swap(
    struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp)
{
    struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->idx && !y->idx || !x->idx && y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
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
 * ‘mtxvector_omp_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_copy(
    struct mtxvector_omp * yomp,
    const struct mtxvector_omp * xomp)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->idx && !y->idx || !x->idx && y->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
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
 * ‘mtxvector_omp_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_omp_sscal(
    float a,
    struct mtxvector_omp * xomp,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_omp_dscal(
    double a,
    struct mtxvector_omp * xomp,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_omp_cscal(
    float a[2],
    struct mtxvector_omp * xomp,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_omp_zscal(
    double a[2],
    struct mtxvector_omp * xomp,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_saxpy(
    float a,
    const struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_daxpy(
    double a,
    const struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_caxpy()’ adds a vector to another one multiplied by
 * a single precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_caxpy(
    float a[2],
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxvector_omp_zaxpy()’ adds a vector to another one multiplied by
 * a double precision floating point complex number, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_zaxpy(
    double a[2],
    const struct mtxvector_omp * x,
    struct mtxvector_omp * y,
    int64_t * num_flops)
{
    return MTX_ERR_NOT_SUPPORTED;
}

/**
 * ‘mtxvector_omp_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_saypx(
    float a,
    struct mtxvector_omp * yomp,
    const struct mtxvector_omp * xomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_daypx(
    double a,
    struct mtxvector_omp * yomp,
    const struct mtxvector_omp * xomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_sdot(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    float * dot,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_ddot(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    double * dot,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_cdotu(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
        return mtxvector_omp_sdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_zdotu(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
        return mtxvector_omp_ddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_cdotc(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
        return mtxvector_omp_sdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_omp_zdotc(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
        return mtxvector_omp_ddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    (*dot)[0] = c0; (*dot)[1] = c1;
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_omp_snrm2(
    const struct mtxvector_omp * xomp,
    float * nrm2,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_omp_dnrm2(
    const struct mtxvector_omp * xomp,
    double * nrm2,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_omp_sasum(
    const struct mtxvector_omp * xomp,
    float * asum,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_omp_dasum(
    const struct mtxvector_omp * xomp,
    double * asum,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
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
 * ‘mtxvector_omp_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the vector is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxvector_omp_iamax(
    const struct mtxvector_omp * xomp,
    int * iamax)
{
    const struct mtxvector_base * x = &xomp->base;
    return mtxvector_base_iamax(x, iamax);
}

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘mtxvector_omp_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_ussdot(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    float * dot,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_sdot(xomp, yomp, dot, num_flops);
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
 * ‘mtxvector_omp_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_usddot(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    double * dot,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_ddot(xomp, yomp, dot, num_flops);
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
 * ‘mtxvector_omp_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uscdotu(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_cdotu(xomp, yomp, dot, num_flops);
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
        return mtxvector_omp_ussdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uszdotu(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_zdotu(xomp, yomp, dot, num_flops);
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
        return mtxvector_omp_usddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uscdotc(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_cdotc(xomp, yomp, dot, num_flops);
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
        return mtxvector_omp_ussdot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_omp_uszdotc(
    const struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_zdotc(xomp, yomp, dot, num_flops);
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
        return mtxvector_omp_usddot(xomp, yomp, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_ussaxpy(
    float alpha,
    const struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_saxpy(alpha, xomp, yomp, num_flops);
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
 * ‘mtxvector_omp_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_usdaxpy(
    double alpha,
    const struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_daxpy(alpha, xomp, yomp, num_flops);
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
 * ‘mtxvector_omp_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_uscaxpy(
    float alpha[2],
    const struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_caxpy(alpha, xomp, yomp, num_flops);
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
 * ‘mtxvector_omp_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_omp_uszaxpy(
    double alpha[2],
    const struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_omp_zaxpy(alpha, xomp, yomp, num_flops);
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
 * ‘mtxvector_omp_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxvector_omp_usga(
    struct mtxvector_omp * xomp,
    const struct mtxvector_omp * yomp)
{
    struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    if (!x->idx) return mtxvector_base_copy(x, y);
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
 * ‘mtxvector_omp_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxvector_omp_usgz(
    struct mtxvector_omp * xomp,
    struct mtxvector_omp * yomp)
{
    struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int mtxvector_omp_ussc(
    struct mtxvector_omp * yomp,
    const struct mtxvector_omp * xomp)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
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
 * ‘mtxvector_omp_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxvector_omp_usscga(
    struct mtxvector_omp * zomp,
    const struct mtxvector_omp * xomp)
{
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * z = &zomp->base;
    if (!x->idx || !z->idx) return MTX_ERR_INVALID_VECTOR_TYPE;
    if (x->field != z->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != z->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != z->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = x->idx;
    struct mtxvector_omp y;
    int err = mtxvector_omp_alloc(&y, x->field, x->precision, x->size);
    if (err) return err;
    err = mtxvector_omp_setzero(&y);
    if (err) { mtxvector_omp_free(&y); return err; }
    err = mtxvector_omp_ussc(&y, xomp);
    if (err) { mtxvector_omp_free(&y); return err; }
    err = mtxvector_omp_usga(zomp, &y);
    if (err) { mtxvector_omp_free(&y); return err; }
    mtxvector_omp_free(&y);
    return MTX_SUCCESS;
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_omp_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_omp_recv()’.
 */
int mtxvector_omp_send(
    const struct mtxvector_omp * x,
    int64_t offset,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    return mtxvector_base_send(
        &x->base, offset, count, recipient, tag, comm, mpierrcode);
}

/**
 * ‘mtxvector_omp_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_omp_send()’.
 */
int mtxvector_omp_recv(
    struct mtxvector_omp * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status,
    int * mpierrcode)
{
    return mtxvector_base_recv(
        &x->base, offset, count, sender, tag, comm, status, mpierrcode);
}

/**
 * ‘mtxvector_omp_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_omp_send()’.
 */
int mtxvector_omp_irecv(
    struct mtxvector_omp * x,
    int64_t offset,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode)
{
    return mtxvector_base_irecv(
        &x->base, offset, count, sender, tag, comm, request, mpierrcode);
}
#endif
#endif
