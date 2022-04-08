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
 * Last modified: 2022-04-08
 *
 * Data structures and routines for basic dense vectors.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/precision.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/vector_array.h>
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
 * ‘mtxvector_base_free()’ frees storage allocated for a vector.
 */
void mtxvector_base_free(
    struct mtxvector_base * x)
{
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
 * ‘mtxvector_base_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxvector_base_alloc_copy(
    struct mtxvector_base * dst,
    const struct mtxvector_base * src)
{
    return mtxvector_base_alloc(dst, src->field, src->precision, src->size);
}

/**
 * ‘mtxvector_base_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxvector_base_init_copy(
    struct mtxvector_base * dst,
    const struct mtxvector_base * src)
{
    int err = mtxvector_base_alloc_copy(dst, src);
    if (err) return err;
    err = mtxvector_base_copy(dst, src);
    if (err) {
        mtxvector_base_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxvector_base_alloc()’ allocates a vector.
 */
int mtxvector_base_alloc(
    struct mtxvector_base * x,
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
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 */
int mtxvector_base_init_real_single(
    struct mtxvector_base * x,
    int64_t size,
    const float * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_real, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.real_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxvector_base_init_real_double(
    struct mtxvector_base * x,
    int64_t size,
    const double * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_real, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.real_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_complex_single()’ allocates and initialises a
 * vector with complex, single precision coefficients.
 */
int mtxvector_base_init_complex_single(
    struct mtxvector_base * x,
    int64_t size,
    const float (* data)[2])
{
    int err = mtxvector_base_alloc(x, mtx_field_complex, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++) {
        x->data.complex_single[k][0] = data[k][0];
        x->data.complex_single[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_complex_double()’ allocates and initialises a
 * vector with complex, double precision coefficients.
 */
int mtxvector_base_init_complex_double(
    struct mtxvector_base * x,
    int64_t size,
    const double (* data)[2])
{
    int err = mtxvector_base_alloc(x, mtx_field_complex, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++) {
        x->data.complex_double[k][0] = data[k][0];
        x->data.complex_double[k][1] = data[k][1];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_integer_single()’ allocates and initialises a
 * vector with integer, single precision coefficients.
 */
int mtxvector_base_init_integer_single(
    struct mtxvector_base * x,
    int64_t size,
    const int32_t * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_integer, mtx_single, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.integer_single[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_integer_double()’ allocates and initialises a
 * vector with integer, double precision coefficients.
 */
int mtxvector_base_init_integer_double(
    struct mtxvector_base * x,
    int64_t size,
    const int64_t * data)
{
    int err = mtxvector_base_alloc(x, mtx_field_integer, mtx_double, size);
    if (err) return err;
    for (int64_t k = 0; k < size; k++)
        x->data.integer_double[k] = data[k];
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_init_pattern()’ allocates and initialises a binary
 * pattern vector, where every entry has a value of one.
 */
int mtxvector_base_init_pattern(
    struct mtxvector_base * x,
    int64_t size)
{
    return mtxvector_base_alloc(x, mtx_field_pattern, mtx_single, size);
}

/*
 * Modifying values
 */

/**
 * ‘mtxvector_base_set_constant_real_single()’ sets every value of a
 * vector equal to a constant, single precision floating point number.
 */
int mtxvector_base_set_constant_real_single(
    struct mtxvector_base * x,
    float a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int mtxvector_base_set_constant_real_double(
    struct mtxvector_base * x,
    double a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_complex_single()’ sets every value of
 * a vector equal to a constant, single precision floating point
 * complex number.
 */
int mtxvector_base_set_constant_complex_single(
    struct mtxvector_base * x,
    float a[2])
{
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_complex_double()’ sets every value of
 * a vector equal to a constant, double precision floating point
 * complex number.
 */
int mtxvector_base_set_constant_complex_double(
    struct mtxvector_base * x,
    double a[2])
{
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a[0];
                x->data.complex_single[k][1] = a[1];
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a[0];
                x->data.complex_double[k][1] = a[1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_integer_single()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_base_set_constant_integer_single(
    struct mtxvector_base * x,
    int32_t a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_set_constant_integer_double()’ sets every value of
 * a vector equal to a constant integer.
 */
int mtxvector_base_set_constant_integer_double(
    struct mtxvector_base * x,
    int64_t a)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.real_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.real_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_single[k][0] = a;
                x->data.complex_single[k][1] = 0;
            }
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++) {
                x->data.complex_double[k][0] = a;
                x->data.complex_double[k][1] = 0;
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_single[k] = a;
        } else if (x->precision == mtx_double) {
            for (int k = 0; k < x->size; k++)
                x->data.integer_double[k] = a;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxvector_base_from_mtxfile()’ converts a vector in Matrix Market
 * format to a vector.
 */
int mtxvector_base_from_mtxfile(
    struct mtxvector_base * x,
    const struct mtxfile * mtxfile)
{
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* TODO: If needed, we could convert from coordinate to array. */
    if (mtxfile->header.format != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            return mtxvector_base_init_real_single(
                x, mtxfile->size.num_rows, mtxfile->data.array_real_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxvector_base_init_real_double(
                x, mtxfile->size.num_rows, mtxfile->data.array_real_double);
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            return mtxvector_base_init_complex_single(
                x, mtxfile->size.num_rows, mtxfile->data.array_complex_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxvector_base_init_complex_double(
                x, mtxfile->size.num_rows, mtxfile->data.array_complex_double);
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            return mtxvector_base_init_integer_single(
                x, mtxfile->size.num_rows, mtxfile->data.array_integer_single);
        } else if (mtxfile->precision == mtx_double) {
            return mtxvector_base_init_integer_double(
                x, mtxfile->size.num_rows, mtxfile->data.array_integer_double);
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        return MTX_ERR_INCOMPATIBLE_MTX_FIELD;
    } else { return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_to_mtxfile()’ converts a vector to a vector in
 * Matrix Market format.
 */
int mtxvector_base_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxvector_base * x,
    enum mtxfileformat mtxfmt)
{
    /* TODO: If needed, we could convert from array to coordinate format. */
    if (mtxfmt != mtxfile_array)
        return MTX_ERR_INCOMPATIBLE_MTX_FORMAT;

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
    return MTX_SUCCESS;
}

/*
 * Partitioning
 */

/**
 * ‘mtxvector_base_partition()’ partitions a vector into blocks
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
int mtxvector_base_partition(
    struct mtxvector * dsts,
    const struct mtxvector_base * src,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile mtxfile;
    err = mtxvector_base_to_mtxfile(&mtxfile, src, mtxfile_array);
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
        dsts[p].type = mtxvector_base;
        err = mtxvector_base_from_mtxfile(
            &dsts[p].storage.base, &dstmtxfiles[p]);
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
 * ‘mtxvector_base_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_base_join(
    struct mtxvector_base * dst,
    const struct mtxvector * srcs,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile * srcmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!srcmtxfiles) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        err = mtxvector_to_mtxfile(&srcmtxfiles[p], &srcs[p], mtxfile_array);
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

    err = mtxvector_base_from_mtxfile(dst, &dstmtxfile);
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
 * ‘mtxvector_base_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_swap(
    struct mtxvector_base * x,
    struct mtxvector_base * y)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < x->size; k++) {
                float z = ydata[k];
                ydata[k] = xdata[k];
                xdata[k] = z;
            }
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_copy(
    struct mtxvector_base * y,
    const struct mtxvector_base * x)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = xdata[k][0];
                ydata[k][1] = xdata[k][1];
            }
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_integer) {
        if (y->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            int32_t * ydata = y->data.integer_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = xdata[k];
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_base_sscal(
    float a,
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxvector_base_dscal(
    double a,
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (a == 1) return MTX_SUCCESS;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * xdata = x->data.real_single;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* xdata)[2] = x->data.complex_single;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            double (* xdata)[2] = x->data.complex_double;
            for (int64_t k = 0; k < x->size; k++) {
                xdata[k][0] *= a;
                xdata[k][1] *= a;
            }
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_base_cscal(
    float a[2],
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
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
 * ‘mtxvector_base_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_base_zscal(
    double a[2],
    struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
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
 * ‘mtxvector_base_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_saxpy(
    float a,
    const struct mtxvector_base * x,
    struct mtxvector_base * y,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
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
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_daxpy(
    double a,
    const struct mtxvector_base * x,
    struct mtxvector_base * y,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] += a*xdata[k][0];
                ydata[k][1] += a*xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
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
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] += a*xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_saypx(
    float a,
    struct mtxvector_base * y,
    const struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
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
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_daypx(
    double a,
    struct mtxvector_base * y,
    const struct mtxvector_base * x,
    int64_t * num_flops)
{
    if (y->field != x->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (y->precision != x->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (y->size != x->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (y->field == mtx_field_real) {
        if (y->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float * ydata = y->data.real_single;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (y->field == mtx_field_complex) {
        if (y->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float (* ydata)[2] = y->data.complex_single;
            for (int64_t k = 0; k < y->size; k++) {
                ydata[k][0] = a*ydata[k][0]+xdata[k][0];
                ydata[k][1] = a*ydata[k][1]+xdata[k][1];
            }
            if (num_flops) *num_flops += 4*y->size;
        } else if (y->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double (* ydata)[2] = y->data.complex_double;
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
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else if (y->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            int64_t * ydata = y->data.integer_double;
            for (int64_t k = 0; k < y->size; k++)
                ydata[k] = a*ydata[k]+xdata[k];
            if (num_flops) *num_flops += 2*y->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_sdot()’ cbaseutes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_sdot(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float * dot,
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            float c = 0;
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
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_ddot()’ cbaseutes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_ddot(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double * dot,
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            const float * ydata = y->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            const double * ydata = y->data.real_double;
            double c = 0;
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
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            const int64_t * ydata = y->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*ydata[k];
            *dot = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_cdotu()’ cbaseutes the product of the transpose of
 * a complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_cdotu(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
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
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_zdotu()’ cbaseutes the product of the transpose of
 * a complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_zdotu(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
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
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] - xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] + xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_cdotc()’ cbaseutes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_cdotc(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    float (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            float c0 = 0, c1 = 0;
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
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_sdot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_zdotc()’ cbaseutes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size.
 */
int mtxvector_base_zdotc(
    const struct mtxvector_base * x,
    const struct mtxvector_base * y,
    double (* dot)[2],
    int64_t * num_flops)
{
    if (x->field != y->field) return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision != y->precision) return MTX_ERR_INCOMPATIBLE_PRECISION;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            const float (* ydata)[2] = y->data.complex_single;
            double c0 = 0, c1 = 0;
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
            for (int64_t k = 0; k < x->size; k++) {
                c0 += xdata[k][0]*ydata[k][0] + xdata[k][1]*ydata[k][1];
                c1 += xdata[k][0]*ydata[k][1] - xdata[k][1]*ydata[k][0];
            }
            (*dot)[0] = c0; (*dot)[1] = c1;
            if (num_flops) *num_flops += 8*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else {
        (*dot)[1] = 0;
        return mtxvector_base_ddot(x, y, &(*dot)[0], num_flops);
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_snrm2()’ cbaseutes the Euclidean norm of a vector
 * in single precision floating point.
 */
int mtxvector_base_snrm2(
    const struct mtxvector_base * x,
    float * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
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
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
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
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_dnrm2()’ cbaseutes the Euclidean norm of a vector
 * in double precision floating point.
 */
int mtxvector_base_dnrm2(
    const struct mtxvector_base * x,
    double * nrm2,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
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
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k][0]*xdata[k][0] + xdata[k][1]*xdata[k][1];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 4*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
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
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += xdata[k]*xdata[k];
            *nrm2 = sqrtf(c);
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_sasum()’ cbaseutes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_sasum(
    const struct mtxvector_base * x,
    float * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            float c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_dasum()’ cbaseutes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed.
 */
int mtxvector_base_dasum(
    const struct mtxvector_base * x,
    double * asum,
    int64_t * num_flops)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            const float (* xdata)[2] = x->data.complex_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += fabs(xdata[k][0]) + fabs(xdata[k][1]);
            *asum = c;
            if (num_flops) *num_flops += 2*x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            const int32_t * xdata = x->data.integer_single;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += abs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            const int64_t * xdata = x->data.integer_double;
            double c = 0;
            for (int64_t k = 0; k < x->size; k++)
                c += llabs(xdata[k]);
            *asum = c;
            if (num_flops) *num_flops += x->size;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_base_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts.
 */
int mtxvector_base_iamax(
    const struct mtxvector_base * x,
    int * iamax)
{
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            const float * xdata = x->data.real_single;
            *iamax = 0;
            float max = x->size > 0 ? fabsf(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabsf(xdata[k])) {
                    max = fabsf(xdata[k]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            *iamax = 0;
            double max = x->size > 0 ? fabs(xdata[0]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
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
            float max = x->size > 0 ? fabsf(xdata[0][0]) + fabsf(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
                if (max < fabsf(xdata[k][0]) + fabsf(xdata[k][1])) {
                    max = fabsf(xdata[k][0]) + fabsf(xdata[k][1]);
                    *iamax = k;
                }
            }
        } else if (x->precision == mtx_double) {
            const double (* xdata)[2] = x->data.complex_double;
            *iamax = 0;
            double max = x->size > 0 ? fabs(xdata[0][0]) + fabs(xdata[0][1]) : 0;
            for (int64_t k = 1; k < x->size; k++) {
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
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }
    return MTX_SUCCESS;
}
