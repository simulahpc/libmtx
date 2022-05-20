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
 * Last modified: 2022-04-26
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
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

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
 * ‘mtxvector_omp_free()’ frees storage allocated for a vector.
 */
void mtxvector_omp_free(
    struct mtxvector_omp * x)
{
    mtxvector_base_free(&x->base);
}

/**
 * ‘mtxvector_omp_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_omp_alloc_copy(
    struct mtxvector_omp * dst,
    const struct mtxvector_omp * src)
{
    return mtxvector_base_alloc_copy(&dst->base, &src->base);
}

/**
 * ‘mtxvector_omp_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_omp_init_copy(
    struct mtxvector_omp * dst,
    const struct mtxvector_omp * src)
{
    return mtxvector_base_init_copy(&dst->base, &src->base);
}

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_omp_alloc()’ allocates a vector.
 */
int mtxvector_omp_alloc(
    struct mtxvector_omp * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int num_threads)
{
    int err = mtxvector_base_alloc(
        &x->base, field, precision, size);
    if (err) return err;
    x->num_threads = num_threads;
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_omp_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_omp_init_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    const float * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_real, mtx_single, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const double * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_real, mtx_double, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const float (* data)[2],
    int num_threads)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_complex, mtx_single, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const double (* data)[2],
    int num_threads)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_complex, mtx_double, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const int32_t * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_integer, mtx_single, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const int64_t * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(
        x, mtx_field_integer, mtx_double, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    int64_t size,
    int num_threads)
{
    return mtxvector_omp_alloc(
        x, mtx_field_pattern, mtx_single, size, num_threads);
}

/*
 * initialise vectors from strided arrays
 */

/**
 * ‘mtxvector_omp_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_omp_init_strided_real_single(
    struct mtxvector_omp * x,
    int64_t size,
    int64_t stride,
    const float * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(x, mtx_field_real, mtx_single, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const double * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(x, mtx_field_real, mtx_double, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const float (* data)[2],
    int num_threads)
{
    int err = mtxvector_omp_alloc(x, mtx_field_complex, mtx_single, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const double (* data)[2],
    int num_threads)
{
    int err = mtxvector_omp_alloc(x, mtx_field_complex, mtx_double, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const int32_t * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(x, mtx_field_integer, mtx_single, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
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
    const int64_t * data,
    int num_threads)
{
    int err = mtxvector_omp_alloc(x, mtx_field_integer, mtx_double, size, num_threads);
    if (err) return err;
    struct mtxvector_base * base = &x->base;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t k = 0; k < size; k++)
        base->data.integer_double[k] = *(const int64_t *) ((const char *) data + k*stride);
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = 0;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = 0;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = 0;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = 0;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = 0;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = 0;
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            #pragma omp parallel for num_threads(num_threads)
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
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
    if (x->base.size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = x->num_threads;
    float * b = x->base.data.real_single;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const float *)((const char *) a + i*stride);
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
    if (x->base.size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = x->num_threads;
    double * b = x->base.data.real_double;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const double *)((const char *) a + i*stride);
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
    if (x->base.size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = x->num_threads;
    float (* b)[2] = x->base.data.complex_single;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < size; i++) {
        b[i][0] = (*(const float (*)[2])((const char *) a + i*stride))[0];
        b[i][1] = (*(const float (*)[2])((const char *) a + i*stride))[1];
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
    if (x->base.size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = x->num_threads;
    double (* b)[2] = x->base.data.complex_double;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < size; i++) {
        b[i][0] = (*(const double (*)[2])((const char *) a + i*stride))[0];
        b[i][1] = (*(const double (*)[2])((const char *) a + i*stride))[1];
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
    if (x->base.size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = x->num_threads;
    int32_t * b = x->base.data.integer_single;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const int32_t *)((const char *) a + i*stride);
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
    if (x->base.size != size) return MTX_ERR_INCOMPATIBLE_SIZE;
    int num_threads = x->num_threads;
    int64_t * b = x->base.data.integer_double;
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < size; i++)
        b[i] = *(const int64_t *)((const char *) a + i*stride);
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
 * ‘mtxvector_omp_partition()’ partitions a vector into blocks
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
int mtxvector_omp_partition(
    struct mtxvector * dsts,
    const struct mtxvector_omp * src,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile mtxfile;
    err = mtxvector_omp_to_mtxfile(&mtxfile, src, 0, NULL, mtxfile_array);
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
        dsts[p].type = mtxvector_omp;
        err = mtxvector_omp_from_mtxfile(
            &dsts[p].storage.omp, &dstmtxfiles[p]);
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
 * ‘mtxvector_omp_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_omp_join(
    struct mtxvector_omp * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile * srcmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!srcmtxfiles) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        err = mtxvector_to_mtxfile(&srcmtxfiles[p], &srcs[p], 0, NULL, mtxfile_array);
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

    err = mtxvector_omp_from_mtxfile(dst, &dstmtxfile);
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                float z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                double z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                float z[2] = {ydata[k][0], ydata[k][1]};
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
                xdata[k][0] = z[0];
                xdata[k][1] = z[1];
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
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
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                int32_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                int64_t z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
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
    int num_threads = xomp->num_threads;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
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
    int num_threads = xomp->num_threads;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t k = 0; k < x->size; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
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
    int num_threads = xomp->num_threads;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t k = 0; k < x->size; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_sdot(xomp, yomp, &(*dot)[0], num_flops);
    }
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_ddot(xomp, yomp, &(*dot)[0], num_flops);
    }
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_sdot(xomp, yomp, &(*dot)[0], num_flops);
    }
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
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_ddot(xomp, yomp, &(*dot)[0], num_flops);
    }
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrt(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrt(c);
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(c);
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrt(c);
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrt(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrt(c);
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    int num_threads = xomp->num_threads;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp,
    float * dot,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp,
    double * dot,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c)
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[idx[k]];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
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
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_ussdot(xpacked, yomp, &(*dot)[0], num_flops);
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
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] - xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] + xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_usddot(xpacked, yomp, &(*dot)[0], num_flops);
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
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            float c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_ussdot(xpacked, yomp, &(*dot)[0], num_flops);
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
    const struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c0 = 0, c1 = 0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:c0,c1)
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[idx[k]][0] + xdata[k][1]*ydata[idx[k]][1];
                c1 += xdata[k][0]*ydata[idx[k]][1] - xdata[k][1]*ydata[idx[k]][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_omp_usddot(xpacked, yomp, &(*dot)[0], num_flops);
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
    struct mtxvector_omp * yomp,
    float alpha,
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
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
    struct mtxvector_omp * yomp,
    double alpha,
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha*xdata[k][0];
                ydata[idx[k]][1] += alpha*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] += alpha*xdata[k];
            if (num_flops) *num_flops += 2*x->size;
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
    struct mtxvector_omp * yomp,
    float alpha[2],
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
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
    struct mtxvector_omp * yomp,
    double alpha[2],
    const struct mtxvector_packed * xpacked,
    int64_t * num_flops)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] += alpha[0]*xdata[k][0] - alpha[1]*xdata[k][1];
                ydata[idx[k]][1] += alpha[0]*xdata[k][1] + alpha[1]*xdata[k][0];
            }
            if (num_flops) *num_flops += 8*x->size;
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
    struct mtxvector_packed * xpacked,
    const struct mtxvector_omp * yomp)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * xdata = x->data.integer_single;
            const int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] = ydata[idx[k]];
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
    struct mtxvector_packed * xpacked,
    struct mtxvector_omp * yomp)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    struct mtxvector_base * x = &xomp->base;
    const struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] = ydata[idx[k]][0];
                xdata[k][1] = ydata[idx[k]][1];
                ydata[idx[k]][0] = 0;
                ydata[idx[k]][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
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
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
            }
        } else if (x->precision == mtx_double) {
            int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k] = ydata[idx[k]];
                ydata[idx[k]] = 0;
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
    const struct mtxvector_packed * xpacked)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    struct mtxvector_base * y = &yomp->base;
    int num_threads = xomp->num_threads < yomp->num_threads
        ? xomp->num_threads : yomp->num_threads;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (xpacked->num_nonzeros != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    const int64_t * idx = xpacked->idx;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] = xdata[k][0];
                ydata[idx[k]][1] = xdata[k][1];
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++) {
                ydata[idx[k]][0] = xdata[k][0];
                ydata[idx[k]][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = 0; k < x->size; k++)
                ydata[idx[k]] = xdata[k];
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
    struct mtxvector_packed * zpacked,
    const struct mtxvector_packed * xpacked)
{
    if (xpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * xomp = &xpacked->x.storage.omp;
    const struct mtxvector_base * x = &xomp->base;
    if (zpacked->x.type != mtxvector_omp) return MTX_ERR_INCOMPATIBLE_VECTOR_TYPE;
    const struct mtxvector_omp * zomp = &zpacked->x.storage.omp;
    const struct mtxvector_base * z = &zomp->base;
    int num_threads = xomp->num_threads < zomp->num_threads
        ? xomp->num_threads : zomp->num_threads;
    if (x->field != z->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != z->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (xpacked->size != zpacked->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    struct mtxvector_omp y;
    int err = mtxvector_omp_alloc(
        &y, x->field, x->precision, xpacked->size, num_threads);
    if (err) return err;
    err = mtxvector_omp_setzero(&y);
    if (err) { mtxvector_omp_free(&y); return err; }
    err = mtxvector_omp_ussc(&y, xpacked);
    if (err) { mtxvector_omp_free(&y); return err; }
    err = mtxvector_omp_usga(zpacked, &y);
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
