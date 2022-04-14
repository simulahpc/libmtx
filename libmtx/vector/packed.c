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
 * Last modified: 2022-04-09
 *
 * Data structures and routines for sparse vectors in packed storage
 * format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
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
 * ‘mtxvector_packed_free()’ frees storage allocated for a vector.
 */
void mtxvector_packed_free(
    struct mtxvector_packed * x)
{
    mtxvector_free(&x->x);
    free(x->idx);
}

/**
 * ‘mtxvector_packed_alloc_copy()’ allocates a copy of a vector
 * without initialising the values.
 */
int mtxvector_packed_alloc_copy(
    struct mtxvector_packed * dst,
    const struct mtxvector_packed * src);

/**
 * ‘mtxvector_packed_init_copy()’ allocates a copy of a vector and
 * also copies the values.
 */
int mtxvector_packed_init_copy(
    struct mtxvector_packed * dst,
    const struct mtxvector_packed * src);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_packed_alloc()’ allocates a sparse vector in packed
 * storage format, where nonzero coefficients are stored in an
 * underlying dense vector of the given type.
 */
int mtxvector_packed_alloc(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros)
{
    x->size = size;
    x->num_nonzeros = num_nonzeros;
    x->idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->idx) return MTX_ERR_ERRNO;
    int err = mtxvector_alloc(&x->x, type, field, precision, num_nonzeros);
    if (err) {
        free(x->idx);
        return err;
    }
    return MTX_SUCCESS;
}

static int mtxvector_packed_init_idx(
    struct mtxvector_packed * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    int err;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (idx[k] < 0 || idx[k] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    x->size = size;
    x->num_nonzeros = num_nonzeros;
    x->idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->idx) return MTX_ERR_ERRNO;
    for (int64_t k = 0; k < num_nonzeros; k++) x->idx[k] = idx[k];
    return MTX_SUCCESS;
}

static int mtxvector_packed_init_strided_idx(
    struct mtxvector_packed * x,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase)
{
    int err;
    x->size = size;
    x->num_nonzeros = num_nonzeros;
    x->idx = malloc(num_nonzeros * sizeof(int64_t));
    if (!x->idx) return MTX_ERR_ERRNO;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        int64_t idxk = (*(int64_t *) ((unsigned char *) idx + k*idxstride)) - idxbase;
        if (idxk < 0 || idxk >= size) {
            free(x->idx);
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }
        x->idx[k] = idxk;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_packed_init_real_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data)
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_real_single(&x->x, type, num_nonzeros, data);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_packed_init_real_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data)
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_real_double(&x->x, type, num_nonzeros, data);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxvector_packed_init_complex_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2])
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_complex_single(&x->x, type, num_nonzeros, data);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxvector_packed_init_complex_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2])
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_complex_double(&x->x, type, num_nonzeros, data);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxvector_packed_init_integer_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data)
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_integer_single(&x->x, type, num_nonzeros, data);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxvector_packed_init_integer_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data)
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_integer_double(&x->x, type, num_nonzeros, data);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxvector_packed_init_pattern(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx)
{
    int err = mtxvector_packed_init_idx(x, size, num_nonzeros, idx);
    if (err) return err;
    err = mtxvector_init_pattern(&x->x, type, num_nonzeros);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxvector_packed_init_strided_real_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase,
    const float * data,
    int64_t datastride)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_strided_real_single(
        &x->x, type, num_nonzeros, data, datastride);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxvector_packed_init_strided_real_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase,
    const double * data,
    int64_t datastride)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_strided_real_double(
        &x->x, type, num_nonzeros, data, datastride);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxvector_packed_init_strided_complex_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase,
    const float (* data)[2],
    int64_t datastride)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_strided_complex_single(
        &x->x, type, num_nonzeros, data, datastride);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxvector_packed_init_strided_complex_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase,
    const double (* data)[2],
    int64_t datastride)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_strided_complex_double(
        &x->x, type, num_nonzeros, data, datastride);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxvector_packed_init_strided_integer_single(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase,
    const int32_t * data,
    int64_t datastride)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_strided_integer_single(
        &x->x, type, num_nonzeros, data, datastride);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxvector_packed_init_strided_integer_double(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase,
    const int64_t * data,
    int64_t datastride)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_strided_integer_double(
        &x->x, type, num_nonzeros, data, datastride);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_init_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxvector_packed_init_strided_pattern(
    struct mtxvector_packed * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    int64_t idxstride,
    int idxbase)
{
    int err = mtxvector_packed_init_strided_idx(
        x, size, num_nonzeros, idx, idxstride, idxbase);
    if (err) return err;
    err = mtxvector_init_pattern(&x->x, type, num_nonzeros);
    if (err) { free(x->idx); return err; }
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxvector_packed_set_constant_real_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point number.
 */
int mtxvector_packed_set_constant_real_single(
    struct mtxvector_packed * x,
    float a)
{
    return mtxvector_set_constant_real_single(&x->x, a);
}

/**
 * ‘mtxvector_packed_set_constant_real_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point number.
 */
int mtxvector_packed_set_constant_real_double(
    struct mtxvector_packed * x,
    double a)
{
    return mtxvector_set_constant_real_double(&x->x, a);
}

/**
 * ‘mtxvector_packed_set_constant_complex_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxvector_packed_set_constant_complex_single(
    struct mtxvector_packed * x,
    float a[2])
{
    return mtxvector_set_constant_complex_single(&x->x, a);
}

/**
 * ‘mtxvector_packed_set_constant_complex_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxvector_packed_set_constant_complex_double(
    struct mtxvector_packed * x,
    double a[2])
{
    return mtxvector_set_constant_complex_double(&x->x, a);
}

/**
 * ‘mtxvector_packed_set_constant_integer_single()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxvector_packed_set_constant_integer_single(
    struct mtxvector_packed * x,
    int32_t a)
{
    return mtxvector_set_constant_integer_single(&x->x, a);
}

/**
 * ‘mtxvector_packed_set_constant_integer_double()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxvector_packed_set_constant_integer_double(
    struct mtxvector_packed * x,
    int64_t a)
{
    return mtxvector_set_constant_integer_double(&x->x, a);
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_packed_from_mtxfile()’ converts from a vector in Matrix
 * Market format.
 */
int mtxvector_packed_from_mtxfile(
    struct mtxvector_packed * x,
    const struct mtxfile * mtxfile,
    enum mtxvectortype type)
{
    int err;
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from array to coordinate. */
    if (mtxfile->header.format != mtxfile_coordinate)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    int64_t size = mtxfile->size.num_rows;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;
    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_real_single * data =
                mtxfile->data.vector_coordinate_real_single;
            err = mtxvector_packed_init_strided_real_single(
                x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1,
                &data[0].a, sizeof(*data));
            if (err) return err;
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_real_double * data =
                mtxfile->data.vector_coordinate_real_double;
            err = mtxvector_packed_init_strided_real_double(
                x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1,
                &data[0].a, sizeof(*data));
            if (err) return err;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_complex_single * data =
                mtxfile->data.vector_coordinate_complex_single;
            err = mtxvector_packed_init_strided_complex_single(
                x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1,
                &data[0].a, sizeof(*data));
            if (err) return err;
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_complex_double * data =
                mtxfile->data.vector_coordinate_complex_double;
            err = mtxvector_packed_init_strided_complex_double(
                x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1,
                &data[0].a, sizeof(*data));
            if (err) return err;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_vector_coordinate_integer_single * data =
                mtxfile->data.vector_coordinate_integer_single;
            err = mtxvector_packed_init_strided_integer_single(
                x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1,
                &data[0].a, sizeof(*data));
            if (err) return err;
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_vector_coordinate_integer_double * data =
                mtxfile->data.vector_coordinate_integer_double;
            err = mtxvector_packed_init_strided_integer_double(
                x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1,
                &data[0].a, sizeof(*data));
            if (err) return err;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        const struct mtxfile_vector_coordinate_pattern * data =
            mtxfile->data.vector_coordinate_pattern;
        err = mtxvector_packed_init_strided_pattern(
            x, type, size, num_nonzeros, &data[0].i, sizeof(*data), 1);
        if (err) return err;
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_to_mtxfile()’ converts to a vector in Matrix
 * Market format.
 */
int mtxvector_packed_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_packed * x,
    enum mtxfileformat mtxfmt)
{
    return mtxvector_to_mtxfile(mtxfile, &x->x, x->size, x->idx, mtxfmt);
}

/*
 * Partitioning
 */

/**
 * ‘mtxvector_packed_partition()’ partitions a vector into blocks
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
int mtxvector_packed_partition(
    struct mtxvector * dsts,
    const struct mtxvector_packed * src,
    const struct mtxpartition * part);

/**
 * ‘mtxvector_packed_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_packed_join(
    struct mtxvector_packed * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxvector_packed_swap()’ swaps values of two vectors,
 * simultaneously performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzero elements.
 */
int mtxvector_packed_swap(
    struct mtxvector_packed * x,
    struct mtxvector_packed * y)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_swap(&x->x, &y->x);
    if (err) return err;
    for (int64_t k = 0; k < x->num_nonzeros; k++) {
        int64_t tmp = y->idx[k];
        y->idx[k] = x->idx[k];
        x->idx[k] = tmp;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_packed_copy(
    struct mtxvector_packed * y,
    const struct mtxvector_packed * x)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_copy(&y->x, &x->x);
    if (err) return err;
    for (int64_t k = 0; k < x->num_nonzeros; k++)
        y->idx[k] = x->idx[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_packed_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_packed_sscal(
    float a,
    struct mtxvector_packed * x,
    int64_t * num_flops)
{
    return mtxvector_sscal(a, &x->x, num_flops);
}

/**
 * ‘mtxvector_packed_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_packed_dscal(
    double a,
    struct mtxvector_packed * x,
    int64_t * num_flops)
{
    return mtxvector_dscal(a, &x->x, num_flops);
}

/**
 * ‘mtxvector_packed_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_packed_cscal(
    float a[2],
    struct mtxvector_packed * x,
    int64_t * num_flops)
{
    return mtxvector_cscal(a, &x->x, num_flops);
}

/**
 * ‘mtxvector_packed_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_packed_zscal(
    double a[2],
    struct mtxvector_packed * x,
    int64_t * num_flops)
{
    return mtxvector_zscal(a, &x->x, num_flops);
}

/**
 * ‘mtxvector_packed_saxpy()’ adds a vector to another one multiplied
 * by a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_saxpy(
    float a,
    const struct mtxvector_packed * x,
    struct mtxvector_packed * y,
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_saxpy(a, &x->x, &y->x, num_flops);
}

/**
 * ‘mtxvector_packed_daxpy()’ adds a vector to another one multiplied
 * by a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_daxpy(
    double a,
    const struct mtxvector_packed * x,
    struct mtxvector_packed * y,
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_daxpy(a, &x->x, &y->x, num_flops);
}

/**
 * ‘mtxvector_packed_saypx()’ multiplies a vector by a single
 * precision floating point scalar and adds another vector, ‘y = a*y +
 * x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_saypx(
    float a,
    struct mtxvector_packed * y,
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_saypx(a, &y->x, &x->x, num_flops);
}

/**
 * ‘mtxvector_packed_daypx()’ multiplies a vector by a double
 * precision floating point scalar and adds another vector, ‘y = a*y +
 * x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_daypx(
    double a,
    struct mtxvector_packed * y,
    const struct mtxvector_packed * x,
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_daypx(a, &y->x, &x->x, num_flops);
}

/**
 * ‘mtxvector_packed_sdot()’ cpackedutes the Euclidean dot product of
 * two vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_sdot(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_sdot(&x->x, &y->x, dot, num_flops);
}

/**
 * ‘mtxvector_packed_ddot()’ cpackedutes the Euclidean dot product of
 * two vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_ddot(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_ddot(&x->x, &y->x, dot, num_flops);
}

/**
 * ‘mtxvector_packed_cdotu()’ cpackedutes the product of the transpose
 * of a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_cdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_cdotu(&x->x, &y->x, dot, num_flops);
}

/**
 * ‘mtxvector_packed_zdotu()’ cpackedutes the product of the transpose
 * of a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_zdotu(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_zdotu(&x->x, &y->x, dot, num_flops);
}

/**
 * ‘mtxvector_packed_cdotc()’ cpackedutes the Euclidean dot product of
 * two complex vectors in single precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_cdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_cdotc(&x->x, &y->x, dot, num_flops);
}

/**
 * ‘mtxvector_packed_zdotc()’ cpackedutes the Euclidean dot product of
 * two complex vectors in double precision floating point, ‘dot :=
 * x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. The offsets of the
 * nonzero entries are assumed to be identical for both vectors,
 * otherwise the results are undefined.
 */
int mtxvector_packed_zdotc(
    const struct mtxvector_packed * x,
    const struct mtxvector_packed * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    /* if (memcmp(x->idx, y->idx, x->num_nonzeros*sizeof(*x->idx)) != 0) return MTX_ERR_INCOMPATIBLE_PATTERN; */
    return mtxvector_zdotc(&x->x, &y->x, dot, num_flops);
}

/**
 * ‘mtxvector_packed_snrm2()’ cpackedutes the Euclidean norm of a
 * vector in single precision floating point.
 */
int mtxvector_packed_snrm2(
    const struct mtxvector_packed * x,
    float * nrm2,
    int64_t * num_flops)
{
    return mtxvector_snrm2(&x->x, nrm2, num_flops);
}

/**
 * ‘mtxvector_packed_dnrm2()’ cpackedutes the Euclidean norm of a
 * vector in double precision floating point.
 */
int mtxvector_packed_dnrm2(
    const struct mtxvector_packed * x,
    double * nrm2,
    int64_t * num_flops)
{
    return mtxvector_dnrm2(&x->x, nrm2, num_flops);
}

/**
 * ‘mtxvector_packed_sasum()’ cpackedutes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_packed_sasum(
    const struct mtxvector_packed * x,
    float * asum,
    int64_t * num_flops)
{
    return mtxvector_sasum(&x->x, asum, num_flops);
}

/**
 * ‘mtxvector_packed_dasum()’ cpackedutes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_packed_dasum(
    const struct mtxvector_packed * x,
    double * asum,
    int64_t * num_flops)
{
    return mtxvector_dasum(&x->x, asum, num_flops);
}

/**
 * ‘mtxvector_packed_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_packed_iamax(
    const struct mtxvector_packed * x,
    int * iamax)
{
    return mtxvector_iamax(&x->x, iamax);
}
