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
 * Data structures and routines for dense vectors with vector
 * operations accelerated by an external BLAS library.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_BLAS
#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/blas.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/vector.h>

#include <cblas.h>

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
 * ‘mtxvector_blas_free()’ frees storage allocated for a vector.
 */
void mtxvector_blas_free(
    struct mtxvector_blas * x)
{
    mtxvector_base_free(&x->base);
}

/**
 * ‘mtxvector_blas_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_blas_alloc_copy(
    struct mtxvector_blas * dst,
    const struct mtxvector_blas * src)
{
    return mtxvector_base_alloc_copy(&dst->base, &src->base);
}

/**
 * ‘mtxvector_blas_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_blas_init_copy(
    struct mtxvector_blas * dst,
    const struct mtxvector_blas * src)
{
    return mtxvector_base_init_copy(&dst->base, &src->base);
}

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_blas_alloc()’ allocates a vector.
 */
int mtxvector_blas_alloc(
    struct mtxvector_blas * x,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size)
{
    return mtxvector_base_alloc(&x->base, field, precision, size);
}

/**
 * ‘mtxvector_blas_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_blas_init_real_single(
    struct mtxvector_blas * x,
    int64_t size,
    const float * data)
{
    return mtxvector_base_init_real_single(&x->base, size, data);
}

/**
 * ‘mtxvector_blas_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_blas_init_real_double(
    struct mtxvector_blas * x,
    int64_t size,
    const double * data)
{
    return mtxvector_base_init_real_double(&x->base, size, data);
}

/**
 * ‘mtxvector_blas_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_blas_init_complex_single(
    struct mtxvector_blas * x,
    int64_t size,
    const float (* data)[2])
{
    return mtxvector_base_init_complex_single(&x->base, size, data);
}

/**
 * ‘mtxvector_blas_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_blas_init_complex_double(
    struct mtxvector_blas * x,
    int64_t size,
    const double (* data)[2])
{
    return mtxvector_base_init_complex_double(&x->base, size, data);
}

/**
 * ‘mtxvector_blas_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_blas_init_integer_single(
    struct mtxvector_blas * x,
    int64_t size,
    const int32_t * data)
{
    return mtxvector_base_init_integer_single(&x->base, size, data);
}

/**
 * ‘mtxvector_blas_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_blas_init_integer_double(
    struct mtxvector_blas * x,
    int64_t size,
    const int64_t * data)
{
    return mtxvector_base_init_integer_double(&x->base, size, data);
}

/**
 * ‘mtxvector_blas_init_pattern()’ allocates and initialises a vector
 * of ones.
 */
int mtxvector_blas_init_pattern(
    struct mtxvector_blas * x,
    int64_t size)
{
    return mtxvector_base_init_pattern(&x->base, size);
}

/*
 * initialise vectors from strided arrays
 */

/**
 * ‘mtxvector_blas_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_blas_init_strided_real_single(
    struct mtxvector_blas * x,
    int64_t size,
    int64_t stride,
    const float * data)
{
    return mtxvector_base_init_strided_real_single(&x->base, size, stride, data);
}

/**
 * ‘mtxvector_blas_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_blas_init_strided_real_double(
    struct mtxvector_blas * x,
    int64_t size,
    int64_t stride,
    const double * data)
{
    return mtxvector_base_init_strided_real_double(&x->base, size, stride, data);
}

/**
 * ‘mtxvector_blas_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_blas_init_strided_complex_single(
    struct mtxvector_blas * x,
    int64_t size,
    int64_t stride,
    const float (* data)[2])
{
    return mtxvector_base_init_strided_complex_single(&x->base, size, stride, data);
}

/**
 * ‘mtxvector_blas_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_blas_init_strided_complex_double(
    struct mtxvector_blas * x,
    int64_t size,
    int64_t stride,
    const double (* data)[2])
{
    return mtxvector_base_init_strided_complex_double(&x->base, size, stride, data);
}

/**
 * ‘mtxvector_blas_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_blas_init_strided_integer_single(
    struct mtxvector_blas * x,
    int64_t size,
    int64_t stride,
    const int32_t * data)
{
    return mtxvector_base_init_strided_integer_single(&x->base, size, stride, data);
}

/**
 * ‘mtxvector_blas_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_blas_init_strided_integer_double(
    struct mtxvector_blas * x,
    int64_t size,
    int64_t stride,
    const int64_t * data)
{
    return mtxvector_base_init_strided_integer_double(&x->base, size, stride, data);
}

/*
 * Modifying values
 */

/**
 * ‘mtxvector_blas_setzero()’ sets every value of a vector to zero.
 */
int mtxvector_blas_setzero(
    struct mtxvector_blas * x)
{
    return mtxvector_base_setzero(&x->base);
}

/**
 * ‘mtxvector_blas_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_blas_set_constant_real_single(
    struct mtxvector_blas * x,
    float a)
{
    return mtxvector_base_set_constant_real_single(&x->base, a);
}

/**
 * ‘mtxvector_blas_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_blas_set_constant_real_double(
    struct mtxvector_blas * x,
    double a)
{
    return mtxvector_base_set_constant_real_double(&x->base, a);
}

/**
 * ‘mtxvector_blas_set_constant_complex_single()’ sets every value of a
 * vector equal to a constant, single precision floating point complex
 * number.
 */
int mtxvector_blas_set_constant_complex_single(
    struct mtxvector_blas * x,
    float a[2])
{
    return mtxvector_base_set_constant_complex_single(&x->base, a);
}

/**
 * ‘mtxvector_blas_set_constant_complex_double()’ sets every value of a
 * vector equal to a constant, double precision floating point complex
 * number.
 */
int mtxvector_blas_set_constant_complex_double(
    struct mtxvector_blas * x,
    double a[2])
{
    return mtxvector_base_set_constant_complex_double(&x->base, a);
}

/**
 * ‘mtxvector_blas_set_constant_integer_single()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxvector_blas_set_constant_integer_single(
    struct mtxvector_blas * x,
    int32_t a)
{
    return mtxvector_base_set_constant_integer_single(&x->base, a);
}

/**
 * ‘mtxvector_blas_set_constant_integer_double()’ sets every value of a
 * vector equal to a constant integer.
 */
int mtxvector_blas_set_constant_integer_double(
    struct mtxvector_blas * x,
    int64_t a)
{
    return mtxvector_base_set_constant_integer_double(&x->base, a);
}

/**
 * ‘mtxvector_blas_set_real_single()’ sets values of a vector based on
 * an array of single precision floating point numbers.
 */
int mtxvector_blas_set_real_single(
    struct mtxvector_blas * x,
    int64_t size,
    int stride,
    const float * a)
{
    return mtxvector_base_set_real_single(&x->base, size, stride, a);
}

/**
 * ‘mtxvector_blas_set_real_double()’ sets values of a vector based on
 * an array of double precision floating point numbers.
 */
int mtxvector_blas_set_real_double(
    struct mtxvector_blas * x,
    int64_t size,
    int stride,
    const double * a)
{
    return mtxvector_base_set_real_double(&x->base, size, stride, a);
}

/**
 * ‘mtxvector_blas_set_complex_single()’ sets values of a vector based
 * on an array of single precision floating point complex numbers.
 */
int mtxvector_blas_set_complex_single(
    struct mtxvector_blas * x,
    int64_t size,
    int stride,
    const float (*a)[2])
{
    return mtxvector_base_set_complex_single(&x->base, size, stride, a);
}

/**
 * ‘mtxvector_blas_set_complex_double()’ sets values of a vector based
 * on an array of double precision floating point complex numbers.
 */
int mtxvector_blas_set_complex_double(
    struct mtxvector_blas * x,
    int64_t size,
    int stride,
    const double (*a)[2])
{
    return mtxvector_base_set_complex_double(&x->base, size, stride, a);
}

/**
 * ‘mtxvector_blas_set_integer_single()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_blas_set_integer_single(
    struct mtxvector_blas * x,
    int64_t size,
    int stride,
    const int32_t * a)
{
    return mtxvector_base_set_integer_single(&x->base, size, stride, a);
}

/**
 * ‘mtxvector_blas_set_integer_double()’ sets values of a vector based
 * on an array of integers.
 */
int mtxvector_blas_set_integer_double(
    struct mtxvector_blas * x,
    int64_t size,
    int stride,
    const int64_t * a)
{
    return mtxvector_base_set_integer_double(&x->base, size, stride, a);
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_blas_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_blas_from_mtxfile(
    struct mtxvector_blas * x,
    const struct mtxfile * mtxfile)
{
    return mtxvector_base_from_mtxfile(&x->base, mtxfile);
}

/**
 * ‘mtxvector_blas_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_blas_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_blas * x,
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
 * ‘mtxvector_blas_partition()’ partitions a vector into blocks
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
int mtxvector_blas_partition(
    struct mtxvector * dsts,
    const struct mtxvector_blas * src,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile mtxfile;
    err = mtxvector_blas_to_mtxfile(&mtxfile, src, 0, NULL, mtxfile_array);
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
        dsts[p].type = mtxvector_blas;
        err = mtxvector_blas_from_mtxfile(
            &dsts[p].storage.blas, &dstmtxfiles[p]);
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
 * ‘mtxvector_blas_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_blas_join(
    struct mtxvector_blas * dst,
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

    err = mtxvector_blas_from_mtxfile(dst, &dstmtxfile);
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
 * ‘mtxvector_blas_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_swap(
    struct mtxvector_blas * xblas,
    struct mtxvector_blas * yblas)
{
    struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_sswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_dswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_cswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_zswap(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_swap(x, y);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_copy(
    struct mtxvector_blas * yblas,
    const struct mtxvector_blas * xblas)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_scopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_dcopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_ccopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_zcopy(y->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        return mtxvector_base_copy(y, x);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_blas_sscal(
    float a,
    struct mtxvector_blas * xblas,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xblas->base;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            cblas_sscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            cblas_dscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            cblas_sscal(2*x->size, a, (float *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            cblas_dscal(2*x->size, a, (double *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_sscal(a, x, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_blas_dscal(
    double a,
    struct mtxvector_blas * xblas,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xblas->base;
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            cblas_sscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            cblas_dscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            cblas_sscal(2*x->size, a, (float *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            cblas_dscal(2*x->size, a, (double *) xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_dscal(a, x, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_blas_cscal(
    float a[2],
    struct mtxvector_blas * xblas,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xblas->base;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        cblas_cscal(x->size, a, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        double az[2] = {a[0], a[1]};
        cblas_zscal(x->size, az, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_blas_zscal(
    double a[2],
    struct mtxvector_blas * xblas,
    int64_t * num_flops)
{
    struct mtxvector_base * x = &xblas->base;
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        float ac[2] = {a[0], a[1]};
        cblas_cscal(x->size, ac, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
        cblas_zscal(x->size, a, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
        if (num_flops) *num_flops += 6*x->size;
    } else { return MTX_ERR_INVALID_PRECISION; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_saxpy(
    float a,
    const struct mtxvector_blas * xblas,
    struct mtxvector_blas * yblas,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_saxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_daxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_saxpy(2*y->size, a, (const float *) xdata, 1, (float *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_daxpy(2*y->size, a, (const double *) xdata, 1, (double *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        return mtxvector_base_saxpy(a, x, y, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_daxpy(
    double a,
    const struct mtxvector_blas * xblas,
    struct mtxvector_blas * yblas,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            cblas_saxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            cblas_daxpy(y->size, a, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            cblas_saxpy(2*y->size, a, (const float *) xdata, 1, (float *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            cblas_daxpy(2*y->size, a, (const double *) xdata, 1, (double *) ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        return mtxvector_base_daxpy(a, x, y, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_saypx(
    float a,
    struct mtxvector_blas * yblas,
    const struct mtxvector_blas * xblas,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    return mtxvector_base_saypx(a, y, x, num_flops);
}

/**
 * ‘mtxvector_blas_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_daypx(
    double a,
    struct mtxvector_blas * yblas,
    const struct mtxvector_blas * xblas,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    struct mtxvector_base * y = &yblas->base;
    return mtxvector_base_daypx(a, y, x, num_flops);
}

/**
 * ‘mtxvector_blas_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_sdot(
    const struct mtxvector_blas * xblas,
    const struct mtxvector_blas * yblas,
    float * dot,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_sdot(x, y, dot, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_ddot(
    const struct mtxvector_blas * xblas,
    const struct mtxvector_blas * yblas,
    double * dot,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            *dot = cblas_sdot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            *dot = cblas_ddot(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_ddot(x, y, dot, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_cdotu(
    const struct mtxvector_blas * xblas,
    const struct mtxvector_blas * yblas,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c[2] = {0, 0};
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_blas_sdot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_zdotu(
    const struct mtxvector_blas * xblas,
    const struct mtxvector_blas * yblas,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c[2] = {0, 0};
            cblas_cdotu_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            cblas_zdotu_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_blas_ddot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_cdotc(
    const struct mtxvector_blas * xblas,
    const struct mtxvector_blas * yblas,
    float (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            double c[2] = {0, 0};
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_blas_sdot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_blas_zdotc(
    const struct mtxvector_blas * xblas,
    const struct mtxvector_blas * yblas,
    double (* dot)[2],
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    const struct mtxvector_base * y = &yblas->base;
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c[2] = {0, 0};
            cblas_cdotc_sub(x->size, xdata, 1, ydata, 1, c);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            (*dot)[0] = c[0]; (*dot)[1] = c[1];
            if (num_flops) *num_flops += 8*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            const double (* ydata)[2] = y->data.complex_double;
            cblas_zdotc_sub(x->size, xdata, 1, ydata, 1, dot);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_blas_ddot(xblas, yblas, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxvector_blas_snrm2(
    const struct mtxvector_blas * xblas,
    float * nrm2,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *nrm2 = cblas_scnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *nrm2 = cblas_dznrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_snrm2(x, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxvector_blas_dnrm2(
    const struct mtxvector_blas * xblas,
    double * nrm2,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *nrm2 = cblas_snrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *nrm2 = cblas_dnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *nrm2 = cblas_scnrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *nrm2 = cblas_dznrm2(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 4*x->size;
        } else {
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_dnrm2(x, nrm2, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_blas_sasum(
    const struct mtxvector_blas * xblas,
    float * asum,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *asum = cblas_sasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *asum = cblas_dasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *asum = cblas_scasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *asum = cblas_dzasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_sasum(x, asum, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_blas_dasum(
    const struct mtxvector_blas * xblas,
    double * asum,
    int64_t * num_flops)
{
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *asum = cblas_sasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *asum = cblas_dasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *asum = cblas_scasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *asum = cblas_dzasum(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_dasum(x, asum, num_flops);
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_blas_iamax()’ finds the index of the first element having
 * the maximum absolute value.  If the vector is complex-valued, then
 * the index points to the first element having the maximum sum of the
 * absolute values of the real and imaginary parts.
 */
int mtxvector_blas_iamax(
    const struct mtxvector_blas * xblas,
    int * iamax)
{
    const struct mtxvector_base * x = &xblas->base;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *iamax = cblas_isamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *iamax = cblas_idamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            *iamax = cblas_icamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *iamax = cblas_izamax(x->size, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        return mtxvector_base_iamax(x, iamax);
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

static struct mtxvector_packed mtxvector_packed_base_from_blas(
    const struct mtxvector_packed * x)
{
    struct mtxvector_packed xbase = {
        .size = x->size,
        .num_nonzeros = x->num_nonzeros,
        .idx = x->idx,
        .x = { .type = mtxvector_base, .storage.base = x->x.storage.blas.base }
    };
    return xbase;
}

/**
 * ‘mtxvector_blas_ussdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_blas_ussdot(
    const struct mtxvector_packed * x,
    const struct mtxvector_blas * y,
    float * dot,
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_ussdot(&xbase, &y->base, dot, num_flops);
}

/**
 * ‘mtxvector_blas_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_blas_usddot(
    const struct mtxvector_packed * x,
    const struct mtxvector_blas * y,
    double * dot,
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_usddot(&xbase, &y->base, dot, num_flops);
}

/**
 * ‘mtxvector_blas_uscdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_blas_uscdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_blas * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_uscdotu(&xbase, &y->base, dot, num_flops);
}

/**
 * ‘mtxvector_blas_uszdotu()’ computes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_blas_uszdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_blas * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_uszdotu(&xbase, &y->base, dot, num_flops);
}

/**
 * ‘mtxvector_blas_uscdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_blas_uscdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_blas * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_uscdotc(&xbase, &y->base, dot, num_flops);
}

/**
 * ‘mtxvector_blas_uszdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. The vector ‘x’ is a sparse vector in packed form. Repeated
 * indices in the packed vector are not allowed, otherwise the result
 * is undefined.
 */
int mtxvector_blas_uszdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_blas * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_uszdotc(&xbase, &y->base, dot, num_flops);
}

/**
 * ‘mtxvector_blas_ussaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_blas_ussaxpy(
    struct mtxvector_blas * y,
    float alpha,
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_ussaxpy(&y->base, alpha, &xbase, num_flops);
}

/**
 * ‘mtxvector_blas_usdaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_blas_usdaxpy(
    struct mtxvector_blas * y,
    double alpha,
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_usdaxpy(&y->base, alpha, &xbase, num_flops);
}

/**
 * ‘mtxvector_blas_uscaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_blas_uscaxpy(
    struct mtxvector_blas * y,
    float alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_uscaxpy(&y->base, alpha, &xbase, num_flops);
}

/**
 * ‘mtxvector_blas_uszaxpy()’ performs a sparse vector update,
 * multiplying a sparse vector ‘x’ in packed form by a scalar ‘alpha’
 * and adding the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size. Repeated indices in the packed vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxvector_blas_uszaxpy(
    struct mtxvector_blas * y,
    double alpha[2],
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_uszaxpy(&y->base, alpha, &xbase, num_flops);
}

/**
 * ‘mtxvector_blas_usga()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are allowed.
 */
int mtxvector_blas_usga(
    struct mtxvector_packed * x,
    const struct mtxvector_blas * y)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_usga(&xbase, &y->base);
}

/**
 * ‘mtxvector_blas_usgz()’ performs a gather operation from a vector
 * ‘y’ into a sparse vector ‘x’ in packed form, while zeroing the
 * values of the source vector ‘y’ that were copied to ‘x’. Repeated
 * indices in the packed vector are allowed.
 */
int mtxvector_blas_usgz(
    struct mtxvector_packed * x,
    struct mtxvector_blas * y)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_usgz(&xbase, &y->base);
}

/**
 * ‘mtxvector_blas_ussc()’ performs a scatter operation to a vector
 * ‘y’ from a sparse vector ‘x’ in packed form. Repeated indices in
 * the packed vector are not allowed, otherwise the result is
 * undefined.
 */
int mtxvector_blas_ussc(
    struct mtxvector_blas * y,
    const struct mtxvector_packed * x)
{
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_ussc(&y->base, &xbase);
}

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxvector_blas_usscga()’ performs a combined scatter-gather
 * operation from a sparse vector ‘x’ in packed form into another
 * sparse vector ‘z’ in packed form. Repeated indices in the packed
 * vector ‘x’ are not allowed, otherwise the result is undefined. They
 * are, however, allowed in the packed vector ‘z’.
 */
int mtxvector_blas_usscga(
    struct mtxvector_packed * z,
    const struct mtxvector_packed * x)
{
    struct mtxvector_packed zbase = mtxvector_packed_base_from_blas(z);
    struct mtxvector_packed xbase = mtxvector_packed_base_from_blas(x);
    return mtxvector_base_usscga(&zbase, &xbase);
}

/*
 * MPI functions
 */

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘mtxvector_blas_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_blas_recv()’.
 */
int mtxvector_blas_send(
    const struct mtxvector_blas * x,
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
 * ‘mtxvector_blas_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_blas_send()’.
 */
int mtxvector_blas_recv(
    struct mtxvector_blas * x,
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
 * ‘mtxvector_blas_irecv()’ performs a non-blocking receive of a
 * vector from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_blas_send()’.
 */
int mtxvector_blas_irecv(
    struct mtxvector_blas * x,
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
