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
 * Data structures for vectors in array format.
 */

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/sort.h>
#include <libmtx/vector/vector_array.h>
#include <libmtx/vector/vector.h>

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
    const struct mtxvector_array * src)
{
    int err = mtxvector_array_alloc_copy(dst, src);
    if (err) return err;
    err = mtxvector_array_copy(dst, src);
    if (err) {
        mtxvector_array_free(dst);
        return err;
    }
    return MTX_SUCCESS;
}

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
    struct mtxfile * mtxfile,
    const struct mtxvector_array * vector,
    enum mtxfileformat mtxfmt)
{
    if (mtxfmt != mtxfile_array)
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
 * Partitioning
 */

/**
 * ‘mtxvector_array_partition()’ partitions a vector into blocks
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
int mtxvector_array_partition(
    struct mtxvector * dsts,
    const struct mtxvector_array * src,
    const struct mtxpartition * part)
{
    int err;
    int num_parts = part ? part->num_parts : 1;
    struct mtxfile mtxfile;
    err = mtxvector_array_to_mtxfile(&mtxfile, src, mtxfile_array);
    if (err) return err;

    struct mtxfile * dstmtxfiles = malloc(sizeof(struct mtxfile) * num_parts);
    if (!dstmtxfiles) return MTX_ERR_ERRNO;
    err = mtxfile_partition(dstmtxfiles, &mtxfile, part, NULL);
    if (err) {
        free(dstmtxfiles);
        return err;
    }

    for (int p = 0; p < num_parts; p++) {
        dsts[p].type = mtxvector_array;
        err = mtxvector_array_from_mtxfile(
            &dsts[p].storage.array, &dstmtxfiles[p]);
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
 * ‘mtxvector_array_join()’ joins together block vectors to form a
 * larger vector.
 *
 * The argument ‘srcs’ is an array of size ‘P’, where ‘P’ is the
 * number of parts in the partitioning (i.e, ‘part->num_parts’).
 */
int mtxvector_array_join(
    struct mtxvector_array * dst,
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

    err = mtxvector_array_from_mtxfile(dst, &dstmtxfile);
    if (err) return err;
    return MTX_SUCCESS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->size; k++)
                ydata[k] = xdata[k];
#endif
        } else if (x->precision == mtx_double) {
            const double * xdata = x->data.real_double;
            double * ydata = y->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dcopy(x->size, xdata, 1, ydata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
#else
            for (int64_t k = 0; k < x->size; k++)
                xdata[k] *= a;
#endif
            if (num_flops) *num_flops += x->size;
        } else if (x->precision == mtx_double) {
            double * xdata = x->data.real_double;
#ifdef LIBMTX_HAVE_BLAS
            cblas_dscal(x->size, a, xdata, 1);
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
 * ‘mtxvector_array_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_array_cscal(
    float a[2],
    struct mtxvector_array * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
        cblas_cscal(x->size, a, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
        double az[2] = {a[0], a[1]};
        cblas_zscal(x->size, az, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->size; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->size;
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxvector_array_zscal(
    double a[2],
    struct mtxvector_array * x,
    int64_t * num_flops)
{
    if (x->field != mtx_field_complex)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    if (x->precision == mtx_single) {
        float (* xdata)[2] = x->data.complex_single;
#ifdef LIBMTX_HAVE_BLAS
        float ac[2] = {a[0], a[1]};
        cblas_cscal(x->size, ac, (float *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->size; k++) {
            float c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            float d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->size;
    } else if (x->precision == mtx_double) {
        double (* xdata)[2] = x->data.complex_double;
#ifdef LIBMTX_HAVE_BLAS
        cblas_zscal(x->size, a, (double *) xdata, 1);
        if (mtxblaserror()) return MTX_ERR_BLAS;
#else
        for (int64_t k = 0; k < x->size; k++) {
            double c = xdata[k][0]*a[0] - xdata[k][1]*a[1];
            double d = xdata[k][0]*a[1] + xdata[k][1]*a[0];
            xdata[k][0] = c;
            xdata[k][1] = d;
        }
#endif
        if (num_flops) *num_flops += 6*x->size;
    } else {
        return MTX_ERR_INVALID_PRECISION;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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
            if (mtxblaserror()) return MTX_ERR_BLAS;
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

/*
 * Sorting
 */

/**
 * ‘mtxvector_array_permute()’ permutes the elements of a vector
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
int mtxvector_array_permute(
    struct mtxvector_array * x,
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
    struct mtxvector_array y;
    int err = mtxvector_array_init_copy(&y, x);
    if (err) return err;

    /* 2. Permute the data. */
    if (x->field == mtx_field_real) {
        if (x->precision == mtx_single) {
            float * dst = x->data.real_single;
            const float * src = y.data.real_single;
            for (int64_t k = 0; k < size; k++)
                dst[perm[k]] = src[offset+k];
        } else if (x->precision == mtx_double) {
            double * dst = x->data.real_double;
            const double * src = y.data.real_double;
            for (int64_t k = 0; k < size; k++)
                dst[perm[k]] = src[offset+k];
        } else {
            mtxvector_array_free(&y);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_complex) {
        if (x->precision == mtx_single) {
            float (* dst)[2] = x->data.complex_single;
            const float (* src)[2] = y.data.complex_single;
            for (int64_t k = 0; k < size; k++) {
                dst[perm[k]][0] = src[offset+k][0];
                dst[perm[k]][1] = src[offset+k][1];
            }
        } else if (x->precision == mtx_double) {
            double (* dst)[2] = x->data.complex_double;
            const double (* src)[2] = y.data.complex_double;
            for (int64_t k = 0; k < size; k++) {
                dst[perm[k]][0] = src[offset+k][0];
                dst[perm[k]][1] = src[offset+k][1];
            }
        } else {
            mtxvector_array_free(&y);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (x->field == mtx_field_integer) {
        if (x->precision == mtx_single) {
            int32_t * dst = x->data.integer_single;
            const int32_t * src = y.data.integer_single;
            for (int64_t k = 0; k < size; k++)
                dst[perm[k]] = src[offset+k];
        } else if (x->precision == mtx_double) {
            int64_t * dst = x->data.integer_double;
            const int64_t * src = y.data.integer_double;
            for (int64_t k = 0; k < size; k++)
                dst[perm[k]] = src[offset+k];
        } else {
            mtxvector_array_free(&y);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else {
        mtxvector_array_free(&y);
        return MTX_ERR_INVALID_MTX_FIELD;
    }
    mtxvector_array_free(&y);
    return MTX_SUCCESS;
}

/**
 * ‘mtxvector_array_sort()’ sorts elements of a vector by the given
 * keys.
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
int mtxvector_array_sort(
    struct mtxvector_array * x,
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
    err = mtxvector_array_permute(x, 0, size, perm);
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
 * ‘mtxvector_array_send()’ sends Matrix Market data lines to another
 * MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘mtxvector_array_recv()’.
 */
int mtxvector_array_send(
    const struct mtxvector_array * data,
    int64_t size,
    int64_t offset,
    int dest,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_recv()’ receives Matrix Market data lines from
 * another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘mtxvector_array_send()’.
 */
int mtxvector_array_recv(
    struct mtxvector_array * data,
    int64_t size,
    int64_t offset,
    int source,
    int tag,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_bcast()’ broadcasts Matrix Market data lines from
 * an MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Bcast()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_array_bcast()’.
 */
int mtxvector_array_bcast(
    struct mtxvector_array * data,
    int64_t size,
    int64_t offset,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_gatherv()’ gathers Matrix Market data lines onto an
 * MPI root process from other processes in a communicator.
 *
 * This is analogous to ‘MPI_Gatherv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_array_gatherv()’.
 */
int mtxvector_array_gatherv(
    const struct mtxvector_array * sendbuf,
    int64_t sendoffset,
    int sendcount,
    struct mtxvector_array * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_scatterv()’ scatters Matrix Market data lines from an
 * MPI root process to other processes in a communicator.
 *
 * This is analogous to ‘MPI_Scatterv()’ and requires every process in
 * the communicator to perform matching calls to
 * ‘mtxvector_array_scatterv()’.
 */
int mtxvector_array_scatterv(
    const struct mtxvector_array * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * displs,
    struct mtxvector_array * recvbuf,
    int64_t recvoffset,
    int recvcount,
    int root,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxvector_array_alltoallv()’ performs an all-to-all exchange of
 * Matrix Market data lines between MPI processes in a communicator.
 *
 * This is analogous to ‘MPI_Alltoallv()’ and requires every process
 * in the communicator to perform matching calls to
 * ‘mtxvector_array_alltoallv()’.
 */
int mtxvector_array_alltoallv(
    const struct mtxvector_array * sendbuf,
    int64_t sendoffset,
    const int * sendcounts,
    const int * senddispls,
    struct mtxvector_array * recvbuf,
    int64_t recvoffset,
    const int * recvcounts,
    const int * recvdispls,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{

    enum mtxfileobject object = mtxfile_vector;
    enum mtxfileformat format = mtxfile_array;
    enum mtxfilefield field = sendbuf->field;
    if (sendbuf->field != recvbuf->field)
        return MTX_ERR_INCOMPATIBLE_FIELD;
    enum mtxprecision precision = sendbuf->precision;
    if (sendbuf->precision != recvbuf->precision)
        return MTX_ERR_INCOMPATIBLE_PRECISION;

    union mtxfiledata senddata;
    union mtxfiledata recvdata;
    if (field == mtx_field_real) {
        field = mtxfile_real;
        if (precision == mtx_single) {
            senddata.array_real_single = sendbuf->data.real_single;
            recvdata.array_real_single = recvbuf->data.real_single;
        } else if (precision == mtx_double) {
            senddata.array_real_double = sendbuf->data.real_double;
            recvdata.array_real_double = recvbuf->data.real_double;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_complex) {
        field = mtxfile_complex;
        if (precision == mtx_single) {
            senddata.array_complex_single = sendbuf->data.complex_single;
            recvdata.array_complex_single = recvbuf->data.complex_single;
        } else if (precision == mtx_double) {
            senddata.array_complex_double = sendbuf->data.complex_double;
            recvdata.array_complex_double = recvbuf->data.complex_double;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtx_field_integer) {
        field = mtxfile_integer;
        if (precision == mtx_single) {
            senddata.array_integer_single = sendbuf->data.integer_single;
            recvdata.array_integer_single = recvbuf->data.integer_single;
        } else if (precision == mtx_double) {
            senddata.array_integer_double = sendbuf->data.integer_double;
            recvdata.array_integer_double = recvbuf->data.integer_double;
        } else { return MTX_ERR_INVALID_PRECISION; }
    } else { return MTX_ERR_INVALID_FIELD; }

    return mtxfiledata_alltoallv(
        &senddata, object, format, field, precision, 0, sendcounts, senddispls,
        &recvdata, 0, recvcounts, recvdispls, comm, disterr);
}
#endif
