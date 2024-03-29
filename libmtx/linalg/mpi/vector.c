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
 * Data structures and routines for distributed sparse vectors in
 * packed form.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/util/mpipartition.h>
#include <libmtx/util/sort.h>
#include <libmtx/linalg/mpi/vector.h>
#include <libmtx/linalg/local/vector.h>

#include <mpi.h>

#include <errno.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Memory management
 */

/**
 * ‘mtxmpivector_free()’ frees storage allocated for a vector.
 */
void mtxmpivector_free(
    struct mtxmpivector * x)
{
    mtxvector_free(&x->xp);
}

static int mtxmpivector_init_comm(
    struct mtxmpivector * x,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    x->comm = comm;
    disterr->mpierrcode = MPI_Comm_size(comm, &x->comm_size);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Comm_rank(comm, &x->rank);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

static int mtxmpivector_init_size(
    struct mtxmpivector * x,
    int64_t size,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    /* check that size is the same on all processes */
    int64_t psize[2] = {-size, size};
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, psize, 2, MPI_INT64_T, MPI_MIN, comm);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (psize[0] != -psize[1]) return MTX_ERR_INCOMPATIBLE_SIZE;
    x->size = size;

    /* sum the number of nonzeros across all processes */
    x->num_nonzeros = num_nonzeros;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &x->num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int mtxmpivector_alloc_copy(
    struct mtxmpivector * dst,
    const struct mtxmpivector * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpivector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int mtxmpivector_init_copy(
    struct mtxmpivector * dst,
    const struct mtxmpivector * src,
    struct mtxdisterror * disterr);

/*
 * Allocation and initialisation
 */

/**
 * ‘mtxmpivector_alloc()’ allocates a sparse vector in packed form,
 * where nonzero coefficients are stored in an underlying dense vector
 * of the given type.
 */
int mtxmpivector_alloc(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    enum mtxfield field,
    enum mtxprecision precision,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_alloc_packed(
        &x->xp, type, field, precision, size, num_nonzeros, idx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_real_single()’ allocates and initialises a
 * vector with real, single precision coefficients.
 *
 * On each process, ‘idx’ and ‘data’ are arrays of length
 * ‘num_nonzeros’, containing the global offsets and values,
 * respectively, of the vector elements stored on the process.
 * ‘num_nonzeros’ may differ from one process to the next. On the
 * other hand, ‘size’ specifies the total number of elements in the
 * entire distributed vector and must be the same on all processes.
 */
int mtxmpivector_init_real_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_real_single(
        &x->xp, type, size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int mtxmpivector_init_real_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_real_double(
        &x->xp, type, size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_complex_single()’ allocates and initialises
 * a vector with complex, single precision coefficients.
 */
int mtxmpivector_init_complex_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_complex_single(
        &x->xp, type, size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_complex_double()’ allocates and initialises
 * a vector with complex, double precision coefficients.
 */
int mtxmpivector_init_complex_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_complex_double(
        &x->xp, type, size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_integer_single()’ allocates and initialises
 * a vector with integer, single precision coefficients.
 */
int mtxmpivector_init_integer_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_integer_single(
        &x->xp, type, size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_integer_double()’ allocates and initialises
 * a vector with integer, double precision coefficients.
 */
int mtxmpivector_init_integer_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_integer_double(
        &x->xp, type, size, num_nonzeros, idx, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxmpivector_init_pattern(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_pattern(
        &x->xp, type, size, num_nonzeros, idx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_strided_real_single()’ allocates and
 * initialises a vector with real, single precision coefficients.
 */
int mtxmpivector_init_strided_real_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_real_single(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_strided_real_double()’ allocates and
 * initialises a vector with real, double precision coefficients.
 */
int mtxmpivector_init_strided_real_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_real_double(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_strided_complex_single()’ allocates and
 * initialises a vector with complex, single precision coefficients.
 */
int mtxmpivector_init_strided_complex_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_complex_single(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_strided_complex_double()’ allocates and
 * initialises a vector with complex, double precision coefficients.
 */
int mtxmpivector_init_strided_complex_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_complex_double(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_strided_integer_single()’ allocates and
 * initialises a vector with integer, single precision coefficients.
 */
int mtxmpivector_init_strided_integer_single(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_integer_single(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_strided_integer_double()’ allocates and
 * initialises a vector with integer, double precision coefficients.
 */
int mtxmpivector_init_strided_integer_double(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    int64_t datastride,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_integer_double(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx, datastride, data);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_init_pattern()’ allocates and initialises a
 * binary pattern vector, where every entry has a value of one.
 */
int mtxmpivector_init_strided_pattern(
    struct mtxmpivector * x,
    enum mtxvectortype type,
    int64_t size,
    int64_t num_nonzeros,
    int64_t idxstride,
    int idxbase,
    const int64_t * idx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;
    err = mtxvector_init_packed_strided_pattern(
        &x->xp, type, size, num_nonzeros, idxstride, idxbase, idx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Modifying values
 */

/**
 * ‘mtxmpivector_setzero()’ sets every nonzero entry of a vector to
 * zero.
 */
int mtxmpivector_setzero(
    struct mtxmpivector * x,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_setzero(&x->xp);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_set_constant_real_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point number.
 */
int mtxmpivector_set_constant_real_single(
    struct mtxmpivector * x,
    float a,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_real_single(&x->xp, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_set_constant_real_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point number.
 */
int mtxmpivector_set_constant_real_double(
    struct mtxmpivector * x,
    double a,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_real_double(&x->xp, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_set_constant_complex_single()’ sets every nonzero
 * entry of a vector equal to a constant, single precision floating
 * point complex number.
 */
int mtxmpivector_set_constant_complex_single(
    struct mtxmpivector * x,
    float a[2],
    struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_complex_single(&x->xp, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_set_constant_complex_double()’ sets every nonzero
 * entry of a vector equal to a constant, double precision floating
 * point complex number.
 */
int mtxmpivector_set_constant_complex_double(
    struct mtxmpivector * x,
    double a[2],
    struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_complex_double(&x->xp, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_set_constant_integer_single()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxmpivector_set_constant_integer_single(
    struct mtxmpivector * x,
    int32_t a,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_integer_single(&x->xp, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_set_constant_integer_double()’ sets every nonzero
 * entry of a vector equal to a constant integer.
 */
int mtxmpivector_set_constant_integer_double(
    struct mtxmpivector * x,
    int64_t a,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_set_constant_integer_double(&x->xp, a);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmpivector_from_mtxfile()’ converts from a vector in Matrix
 * Market format.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each process.
 */
int mtxmpivector_from_mtxfile(
    struct mtxmpivector * x,
    const struct mtxfile * mtxfile,
    enum mtxvectortype type,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int err = mtxmpivector_init_comm(x, comm, disterr);
    if (err) return err;
    int comm_size = x->comm_size;
    int rank = x->rank;

    /* broadcast the header of the Matrix Market file */
    struct mtxfileheader mtxheader;
    if (rank == root) mtxheader = mtxfile->header;
    err = mtxfileheader_bcast(&mtxheader, root, comm, disterr);
    if (err) return err;
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    /* broadcast the size of the Matrix Market file */
    struct mtxfilesize mtxsize;
    if (rank == root) mtxsize = mtxfile->size;
    err = mtxfilesize_bcast(&mtxsize, root, comm, disterr);
    if (err) return err;

    /* divide rows or nonzeros into equal-sized blocks */
    int64_t size = mtxsize.num_rows;
    int64_t num_nonzeros;
    if (mtxfile->header.format == mtxfile_array) {
        num_nonzeros = mtxsize.num_rows / comm_size
            + (rank < (mtxsize.num_rows % comm_size) ? 1 : 0);
    } else if (mtxfile->header.format == mtxfile_coordinate) {
        num_nonzeros = mtxsize.num_nonzeros / comm_size
            + (rank < (mtxsize.num_nonzeros % comm_size) ? 1 : 0);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    err = mtxmpivector_init_size(x, size, num_nonzeros, comm, disterr);
    if (err) return err;

    int64_t offset = 0;
    for (int p = 0; p < comm_size; p++) {
        int64_t num_nonzeros;
        if (mtxfile->header.format == mtxfile_array) {
            num_nonzeros = mtxsize.num_rows / comm_size
                + (p < (mtxsize.num_rows % comm_size) ? 1 : 0);
        } else if (mtxfile->header.format == mtxfile_coordinate) {
            num_nonzeros = mtxsize.num_nonzeros / comm_size
                + (p < (mtxsize.num_nonzeros % comm_size) ? 1 : 0);
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }

        /* extract a matrix market file for the current process */
        struct mtxfile sendmtxfile;
        if (rank == root) {
            struct mtxfilesize sendsize = mtxfile->size;
            if (mtxfile->header.format == mtxfile_array)
                sendsize.num_rows = num_nonzeros;
            else if (mtxfile->header.format == mtxfile_coordinate)
                sendsize.num_nonzeros = num_nonzeros;
            else err = MTX_ERR_INVALID_MTX_FORMAT;
            err = err ? err : mtxfile_alloc(
                &sendmtxfile, &mtxfile->header, &mtxfile->comments,
                &sendsize, mtxfile->precision);
            err = err ? err : mtxfiledata_copy(
                &sendmtxfile.data, &mtxfile->data,
                sendmtxfile.header.object, sendmtxfile.header.format,
                sendmtxfile.header.field, sendmtxfile.precision,
                num_nonzeros, 0, offset);
        }
        if (rank == root && p != root) {
            /* send from the root process */
            err = err ? err : mtxfile_send(&sendmtxfile, p, 0, comm, disterr);
            mtxfile_free(&sendmtxfile);
        } else if (rank != root && rank == p) {
            /* receive from the root process */
            struct mtxfile recvmtxfile;
            err = err ? err : mtxfile_recv(&recvmtxfile, root, 0, comm, disterr);
            err = err ? err : mtxvector_from_mtxfile(
                &x->xp, &recvmtxfile, type);
            mtxfile_free(&recvmtxfile);
        } else if (rank == root && p == root) {
            err = err ? err : mtxvector_from_mtxfile(
                &x->xp, &sendmtxfile, type);
            mtxfile_free(&sendmtxfile);
        }
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        offset += num_nonzeros;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_to_mtxfile()’ converts to a vector in Matrix Market
 * format.
 */
int mtxmpivector_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmpivector * x,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    enum mtxfield field;
    err = mtxvector_field(&x->xp, &field);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxprecision precision;
    err = mtxvector_precision(&x->xp, &precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    struct mtxfileheader mtxheader;
    mtxheader.object = mtxfile_vector;
    mtxheader.format = mtxfmt;
    if (field == mtx_field_real) mtxheader.field = mtxfile_real;
    else if (field == mtx_field_complex) mtxheader.field = mtxfile_complex;
    else if (field == mtx_field_integer) mtxheader.field = mtxfile_integer;
    else if (field == mtx_field_pattern) mtxheader.field = mtxfile_pattern;
    else { return MTX_ERR_INVALID_FIELD; }
    mtxheader.symmetry = mtxfile_general;

    struct mtxfilesize mtxsize;
    mtxsize.num_rows = x->size;
    mtxsize.num_columns = -1;
    if (mtxfmt == mtxfile_array) {
        mtxsize.num_nonzeros = -1;
    } else if (mtxfmt == mtxfile_coordinate) {
        mtxsize.num_nonzeros = x->num_nonzeros;
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    if (x->rank == root)
        err = mtxfile_alloc(mtxfile, &mtxheader, NULL, &mtxsize, precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int64_t offset = 0;
    for (int p = 0; p < x->comm_size; p++) {
        if (x->rank == root && p != root) {
            /* receive from the root process */
            struct mtxfile recvmtxfile;
            err = err ? err : mtxfile_recv(&recvmtxfile, p, 0, x->comm, disterr);
            int64_t num_nonzeros = 0;
            if (mtxfile->header.format == mtxfile_array) {
                num_nonzeros = recvmtxfile.size.num_rows;
            } else if (mtxfile->header.format == mtxfile_coordinate) {
                num_nonzeros = recvmtxfile.size.num_nonzeros;
            } else { err = MTX_ERR_INVALID_MTX_FORMAT; }
            err = err ? err : mtxfiledata_copy(
                &mtxfile->data, &recvmtxfile.data,
                recvmtxfile.header.object, recvmtxfile.header.format,
                recvmtxfile.header.field, recvmtxfile.precision,
                num_nonzeros, offset, 0);
            mtxfile_free(&recvmtxfile);
            offset += num_nonzeros;
        } else if (x->rank != root && x->rank == p) {
            /* send to the root process */
            struct mtxfile sendmtxfile;
            err = mtxvector_to_mtxfile(&sendmtxfile, &x->xp, 0, NULL, mtxfmt);
            err = err ? err : mtxfile_send(&sendmtxfile, root, 0, x->comm, disterr);
            mtxfile_free(&sendmtxfile);
        } else if (x->rank == root && p == root) {
            struct mtxfile sendmtxfile;
            err = mtxvector_to_mtxfile(&sendmtxfile, &x->xp, 0, NULL, mtxfmt);
            int64_t num_nonzeros;
            err = err ? err : mtxvector_num_nonzeros(&x->xp, &num_nonzeros);
            err = err ? err : mtxfiledata_copy(
                &mtxfile->data, &sendmtxfile.data,
                sendmtxfile.header.object, sendmtxfile.header.format,
                sendmtxfile.header.field, sendmtxfile.precision,
                num_nonzeros, offset, 0);
            mtxfile_free(&sendmtxfile);
            offset += num_nonzeros;
        }
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_from_mtxdistfile()’ converts from a vector in
 * Matrix Market format that is distributed among multiple processes.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxvector’ on each process.
 */
int mtxmpivector_from_mtxdistfile(
    struct mtxmpivector * x,
    const struct mtxdistfile * mtxdistfile,
    enum mtxvectortype type,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err;
    if (mtxdistfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    if (mtxdistfile->header.format == mtxfile_array) {
        int64_t size = mtxdistfile->size.num_rows;
        int64_t num_nonzeros = mtxdistfile->localdatasize;
        const int64_t * idx = mtxdistfile->idx;
        if (mtxdistfile->header.field == mtxfile_real) {
            if (mtxdistfile->precision == mtx_single) {
                const float * data = mtxdistfile->data.array_real_single;
                err = mtxmpivector_init_real_single(
                    x, type, size, num_nonzeros, idx, data, comm, disterr);
                if (err) { return err; }
            } else if (mtxdistfile->precision == mtx_double) {
                const double * data = mtxdistfile->data.array_real_double;
                err = mtxmpivector_init_real_double(
                    x, type, size, num_nonzeros, idx, data, comm, disterr);
                if (err) { return err; }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_complex) {
            if (mtxdistfile->precision == mtx_single) {
                const float (* data)[2] = mtxdistfile->data.array_complex_single;
                err = mtxmpivector_init_complex_single(
                    x, type, size, num_nonzeros, idx, data, comm, disterr);
                if (err) { return err; }
            } else if (mtxdistfile->precision == mtx_double) {
                const double (* data)[2] = mtxdistfile->data.array_complex_double;
                err = mtxmpivector_init_complex_double(
                    x, type, size, num_nonzeros, idx, data, comm, disterr);
                if (err) { return err; }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_integer) {
            if (mtxdistfile->precision == mtx_single) {
                const int32_t * data = mtxdistfile->data.array_integer_single;
                err = mtxmpivector_init_integer_single(
                    x, type, size, num_nonzeros, idx, data, comm, disterr);
                if (err) { return err; }
            } else if (mtxdistfile->precision == mtx_double) {
                const int64_t * data = mtxdistfile->data.array_integer_double;
                err = mtxmpivector_init_integer_double(
                    x, type, size, num_nonzeros, idx, data, comm, disterr);
                if (err) { return err; }
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_pattern) {
            err = mtxmpivector_init_pattern(
                x, type, size, num_nonzeros, idx, comm, disterr);
            if (err) { return err; }
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxdistfile->header.format == mtxfile_coordinate) {
        int64_t size = mtxdistfile->size.num_rows;
        int64_t num_nonzeros = mtxdistfile->localdatasize;
        if (mtxdistfile->header.field == mtxfile_real) {
            if (mtxdistfile->precision == mtx_single) {
                const struct mtxfile_vector_coordinate_real_single * data =
                    mtxdistfile->data.vector_coordinate_real_single;
                err = mtxmpivector_init_strided_real_single(
                    x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else if (mtxdistfile->precision == mtx_double) {
                const struct mtxfile_vector_coordinate_real_double * data =
                    mtxdistfile->data.vector_coordinate_real_double;
                err = mtxmpivector_init_strided_real_double(
                    x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_complex) {
            if (mtxdistfile->precision == mtx_single) {
                const struct mtxfile_vector_coordinate_complex_single * data =
                    mtxdistfile->data.vector_coordinate_complex_single;
                err = mtxmpivector_init_strided_complex_single(
                    x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else if (mtxdistfile->precision == mtx_double) {
                const struct mtxfile_vector_coordinate_complex_double * data =
                    mtxdistfile->data.vector_coordinate_complex_double;
                err = mtxmpivector_init_strided_complex_double(
                    x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_integer) {
            if (mtxdistfile->precision == mtx_single) {
                const struct mtxfile_vector_coordinate_integer_single * data =
                    mtxdistfile->data.vector_coordinate_integer_single;
                err = mtxmpivector_init_strided_integer_single(
                    x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else if (mtxdistfile->precision == mtx_double) {
                const struct mtxfile_vector_coordinate_integer_double * data =
                    mtxdistfile->data.vector_coordinate_integer_double;
                err = mtxmpivector_init_strided_integer_double(
                    x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_pattern) {
            const struct mtxfile_vector_coordinate_pattern * data =
                mtxdistfile->data.vector_coordinate_pattern;
            err = mtxmpivector_init_strided_pattern(
                x, type, size, num_nonzeros, sizeof(*data), 1, &data[0].i,
                comm, disterr);
            if (err) return err;
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_to_mtxdistfile()’ converts to a vector in Matrix
 * Market format that is distributed among multiple processes.
 */
int mtxmpivector_to_mtxdistfile(
    struct mtxdistfile * mtxdistfile,
    const struct mtxmpivector * x,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr)
{
    enum mtxfield field;
    int err = mtxvector_field(&x->xp, &field);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxprecision precision;
    err = mtxvector_precision(&x->xp, &precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t num_nonzeros;
    err = mtxvector_num_nonzeros(&x->xp, &num_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    const int64_t * idx;
    err = mtxvector_idx(&x->xp, (int64_t **) &idx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    struct mtxfileheader mtxheader;
    mtxheader.object = mtxfile_vector;
    mtxheader.format = mtxfmt;
    if (field == mtx_field_real) mtxheader.field = mtxfile_real;
    else if (field == mtx_field_complex) mtxheader.field = mtxfile_complex;
    else if (field == mtx_field_integer) mtxheader.field = mtxfile_integer;
    else if (field == mtx_field_pattern) mtxheader.field = mtxfile_pattern;
    else { return MTX_ERR_INVALID_FIELD; }
    mtxheader.symmetry = mtxfile_general;

    struct mtxfilesize mtxsize;
    mtxsize.num_rows = x->size;
    mtxsize.num_columns = -1;
    if (mtxfmt == mtxfile_array) {
        mtxsize.num_nonzeros = -1;
    } else if (mtxfmt == mtxfile_coordinate) {
        mtxsize.num_nonzeros = x->num_nonzeros;
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    err = mtxdistfile_alloc(
        mtxdistfile, &mtxheader, NULL, &mtxsize, precision,
        num_nonzeros, idx, x->comm, disterr);
    if (err) return err;

    struct mtxfile mtxfile;
    err = mtxvector_to_mtxfile(&mtxfile, &x->xp, 0, NULL, mtxfmt);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    err = mtxfiledata_copy(
        &mtxdistfile->data, &mtxfile.data,
        mtxfile.header.object, mtxfile.header.format,
        mtxfile.header.field, mtxfile.precision,
        num_nonzeros, 0, 0);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfile_free(&mtxfile);
        mtxdistfile_free(mtxdistfile);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/*
 * I/O operations
 */

/**
 * ‘mtxmpivector_fwrite()’ writes a distributed vector to a single
 * stream that is shared by every process in the communicator. The
 * output is written in Matrix Market format.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’ or ‘complex’, then the format specifiers '%e', '%E',
 * '%f', '%F', '%g' or '%G' may be used. If the field is ‘integer’,
 * then the format specifier must be '%d'. The format string is
 * ignored if the field is ‘pattern’. Field width and precision may be
 * specified (e.g., "%3.1f"), but variable field width and precision
 * (e.g., "%*.*f"), as well as length modifiers (e.g., "%Lf") are not
 * allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * Note that only the specified ‘root’ process will print anything to
 * the stream. Other processes will therefore send their part of the
 * distributed data to the root process for printing.
 *
 * This function performs collective communication and therefore
 * requires every process in the communicator to perform matching
 * calls to the function.
 */
int mtxmpivector_fwrite(
    const struct mtxmpivector * x,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr)
{
    struct mtxdistfile dst;
    int err = mtxmpivector_to_mtxdistfile(&dst, x, mtxfmt, disterr);
    if (err) return err;
    err = mtxdistfile_fwrite(&dst, f, fmt, bytes_written, root, disterr);
    if (err) { mtxdistfile_free(&dst); return err; }
    mtxdistfile_free(&dst);
    return MTX_SUCCESS;
}

/*
 * partitioning
 */

/**
 * ‘mtxmpivector_split()’ splits a vector into multiple vectors
 * according to a given assignment of parts to each vector element.
 *
 * The partitioning of the vector elements is specified by the array
 * ‘parts’. The length of the ‘parts’ array is given by ‘size’, which
 * must match the size of the vector ‘src’. Each entry in the array is
 * an integer in the range ‘[0, num_parts)’ designating the part to
 * which the corresponding vector element belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxmpivector’. If successful, then ‘dsts[p]’
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
 * The caller is responsible for calling ‘mtxmpivector_free()’ to
 * free storage allocated for each vector in the ‘dsts’ array.
 */
int mtxmpivector_split(
    int num_parts,
    struct mtxmpivector ** dsts,
    const struct mtxmpivector * src,
    int64_t size,
    int * parts,
    int64_t * invperm,
    struct mtxdisterror * disterr)
{
    struct mtxvector ** packeddsts = malloc(
        num_parts * sizeof(struct mtxvector *));
    if (!packeddsts) return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) packeddsts[p] = &dsts[p]->xp;
    int err = mtxvector_split(
        num_parts, packeddsts, &src->xp, size, parts, invperm);
    if (err) { free(packeddsts); return err; }
    free(packeddsts);
    for (int p = 0; p < num_parts; p++) {
        dsts[p]->comm = src->comm;
        dsts[p]->comm_size = src->comm_size;
        dsts[p]->rank = src->rank;
        int64_t size;
        err = mtxvector_size(&dsts[p]->xp, &size);
        if (err) {
            for (int q = 0; q < num_parts; q++) mtxvector_free(&dsts[q]->xp);
            return err;
        }
        int64_t num_nonzeros;
        err = mtxvector_num_nonzeros(&dsts[p]->xp, &num_nonzeros);
        if (err) {
            for (int q = 0; q < num_parts; q++) mtxvector_free(&dsts[q]->xp);
            return err;
        }
        err = mtxmpivector_init_size(
            dsts[p], size, num_nonzeros, dsts[p]->comm, disterr);
        if (err) {
            for (int q = 0; q < num_parts; q++) mtxvector_free(&dsts[q]->xp);
            return err;
        }
    }
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmpivector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both vectors must also have the same number of
 * nonzero elements on that process.
 */
int mtxmpivector_swap(
    struct mtxmpivector * x,
    struct mtxmpivector * y,
    struct mtxdisterror * disterr)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_swap(&x->xp, &y->xp);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both vectors must also have the same number of
 * nonzero elements on that process.
 */
int mtxmpivector_copy(
    struct mtxmpivector * y,
    const struct mtxmpivector * x,
    struct mtxdisterror * disterr)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_copy(&y->xp, &x->xp);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_sscal()’ scales a vector by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmpivector_sscal(
    float a,
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_sscal(a, &x->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_dscal()’ scales a vector by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmpivector_dscal(
    double a,
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_dscal(a, &x->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_cscal()’ scales a vector by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmpivector_cscal(
    float a[2],
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_cscal(a, &x->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_zscal()’ scales a vector by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmpivector_zscal(
    double a[2],
    struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_zscal(a, &x->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_saxpy()’ adds a vector to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the packed vectors are
 * allowed.
 */
int mtxmpivector_saxpy(
    float a,
    const struct mtxmpivector * x,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_saxpy(a, &x->xp, &y->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_daxpy()’ adds a vector to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the packed vectors are
 * allowed.
 */
int mtxmpivector_daxpy(
    double a,
    const struct mtxmpivector * x,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_daxpy(a, &x->xp, &y->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_saypx()’ multiplies a vector by a single precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the packed vectors are
 * allowed.
 */
int mtxmpivector_saypx(
    float a,
    struct mtxmpivector * y,
    const struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_saypx(a, &y->xp, &x->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. However, repeated indices in the packed vectors are
 * allowed.
 */
int mtxmpivector_daypx(
    double a,
    struct mtxmpivector * y,
    const struct mtxmpivector * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxvector_daypx(a, &y->xp, &x->xp, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_sdot()’ computes the Euclidean dot product of two
 * vectors in single precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpivector_sdot(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    float * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxvector_sdot(&x->xp, &y->xp, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpivector_ddot(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    double * dot,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxvector_ddot(&x->xp, &y->xp, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_cdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpivector_cdotu(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxvector_cdotu(&x->xp, &y->xp, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_zdotu()’ computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpivector_zdotu(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxvector_zdotu(&x->xp, &y->xp, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_cdotc()’ computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpivector_cdotc(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    float (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxvector_cdotc(&x->xp, &y->xp, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_zdotc()’ computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 *
 * The vectors ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both vectors must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both vectors, otherwise the results are
 * undefined. Moreover, repeated indices in the dist vector are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpivector_zdotc(
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    double (* dot)[2],
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, y->comm, &result);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    err = result != MPI_IDENT ? MTX_ERR_INCOMPATIBLE_MPI_COMM : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (x->size != y->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxvector_zdotc(&x->xp, &y->xp, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_snrm2()’ computes the Euclidean norm of a vector in
 * single precision floating point. Repeated indices in the dist
 * vector are not allowed, otherwise the result is undefined.
 */
int mtxmpivector_snrm2(
    const struct mtxmpivector * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    float dot[2] = {0.0f, 0.0f};
    int err = mtxvector_cdotc(&x->xp, &x->xp, &dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(dot[0]);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point. Repeated indices in the dist
 * vector are not allowed, otherwise the result is undefined.
 */
int mtxmpivector_dnrm2(
    const struct mtxmpivector * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    double dot[2] = {0.0, 0.0};
    int err = mtxvector_zdotc(&x->xp, &x->xp, &dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(dot[0]);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_sasum()’ computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist vector are not allowed, otherwise the result is undefined.
 */
int mtxmpivector_sasum(
    const struct mtxmpivector * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_sasum(&x->xp, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_dasum()’ computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.  If the
 * vector is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist vector are not allowed, otherwise the result is undefined.
 */
int mtxmpivector_dasum(
    const struct mtxmpivector * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxvector_dasum(&x->xp, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the vector is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts. Repeated indices in the dist vector are not allowed,
 * otherwise the result is undefined.
 */
int mtxmpivector_iamax(
    const struct mtxmpivector * x,
    int * iamax,
    struct mtxdisterror * disterr);

/*
 * Level 1 BLAS-like extensions
 */

/**
 * ‘mtxmpivector_usscga()’ performs a combined scatter-gather
 * operation from a distributed sparse vector ‘x’ in packed form into
 * another distributed sparse vector ‘z’ in packed form. The vectors
 * ‘x’ and ‘z’ must have the same field, precision and size. Repeated
 * indices in the packed vector ‘x’ are not allowed, otherwise the
 * result is undefined. They are, however, allowed in the packed
 * vector ‘z’.
 */
int mtxmpivector_usscga(
    struct mtxmpivector * z,
    const struct mtxmpivector * x,
    struct mtxdisterror * disterr)
{
    struct mtxmpivector_usscga usscga;
    int err = mtxmpivector_usscga_init(&usscga, z, x, disterr);
    if (err) return err;
    err = mtxmpivector_usscga_start(&usscga, disterr);
    if (err) { mtxmpivector_usscga_free(&usscga); return err; }
    err = mtxmpivector_usscga_wait(&usscga, disterr);
    if (err) { mtxmpivector_usscga_free(&usscga); return err; }
    mtxmpivector_usscga_free(&usscga);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_usscga_impl’ is a data structure for a persistent,
 * asynchronous, combined scatter-gather operation.
 */
struct mtxmpivector_usscga_impl
{
    MPI_Comm comm;
    int commsize;
    int rank;
    enum mtxfield field;
    enum mtxprecision precision;
    int64_t * zperm;
    int64_t idxsrcrankstart;
    int nrecvranks;
    int * recvranks;
    int * recvcounts;
    int * rdispls;
    int nsendranks;
    int * sendranks;
    int * sendcounts;
    int * sdispls;
    struct mtxvector sendbuf;
    struct mtxvector recvbuf;

    /**
     * ‘req’ is an array of length ‘comm_size’, containing MPI
     * requests used for non-blocking communication.
     */
    MPI_Request * req;
};

/**
 * ‘mtxmpivector_usscga_init()’ allocates data structures for a
 * persistent, combined scatter-gather operation.
 *
 * This is used in cases where the combined scatter-gather operation
 * is performed repeatedly, since the setup phase only needs to be
 * carried out once.
 */
int mtxmpivector_usscga_init(
    struct mtxmpivector_usscga * usscga,
    struct mtxmpivector * z,
    const struct mtxmpivector * x,
    struct mtxdisterror * disterr)
{
    int err;
    int result;
    disterr->mpierrcode = MPI_Comm_compare(x->comm, z->comm, &result);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (result != MPI_IDENT) return MTX_ERR_INCOMPATIBLE_MPI_COMM;
    if (x->size != z->size) return MTX_ERR_INCOMPATIBLE_SIZE;
    MPI_Comm comm = x->comm;
    int comm_size = x->comm_size;
    int rank = x->rank;
    int64_t size = x->size;
    int apsize = (size+comm_size-1) / comm_size;

    enum mtxfield field;
    err = mtxvector_field(&x->xp, &field);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxprecision precision;
    err = mtxvector_precision(&x->xp, &precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t xnum_nonzeros;
    err = mtxvector_num_nonzeros(&x->xp, &xnum_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t znum_nonzeros;
    err = mtxvector_num_nonzeros(&z->xp, &znum_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    const int64_t * xidx;
    err = mtxvector_idx(&x->xp, (int64_t **) &xidx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    const int64_t * zidx;
    err = mtxvector_idx(&z->xp, (int64_t **) &zidx);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /*
     * The algorithm proceeds in two phases, where the first phase
     * involves exchanging metadata, whereas the second phase consists
     * of exchanging the actual data itself.
     *
     * More precisely, each process begins by knowing which elements
     * of the input vector ‘x’ that it owns, and also which elements
     * of the input vector that it must receive from other processes
     * to populate its portion of the output vector ‘z’. However, a
     * process does not know at this point which elements of the input
     * vector it must send to other processes. Therefore, the first
     * phase is designed to inform each process about which of its
     * input vector elements it must send to other processes.
     *
     * Thereafter, every process will know which input vector elements
     * to receive and where they will be received from, and they will
     * also know which of their input vector elements to send and
     * where they must be sent to. The data exchange can therefore be
     * carried out with appropriate calls to ‘MPI_Send’ and
     * ‘MPI_Recv’.
     */

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "xidx=[");
            for (int64_t i = 0; i < xnum_nonzeros; i++)
                fprintf(stderr, " %"PRId64, xidx[i]);
            fprintf(stderr, "], zidx=[");
            for (int64_t i = 0; i < znum_nonzeros; i++)
                fprintf(stderr, " %"PRId64, zidx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 0: Prepare data structures for the assumed partition
     * strategy for the input vector.
     */

    int * apownerrank = malloc(apsize * sizeof(int));
    err = !apownerrank ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int * apowneridx = malloc(apsize * sizeof(int));
    err = !apowneridx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(apownerrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = assumedpartition_write(
        size, xnum_nonzeros, xidx, apsize,
        apownerrank, apowneridx, comm, disterr);
    if (err) {
        free(apowneridx); free(apownerrank);
        return err;
    }

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "apownerrank=[");
            for (int64_t i = 0; i < apsize; i++)
                fprintf(stderr, " %d", apownerrank[i]);
            fprintf(stderr, "], apowneridx=[");
            for (int64_t i = 0; i < apsize; i++)
                fprintf(stderr, " %d", apowneridx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 1: For each element of the output array, find which
     * process owns the corresponding element of the input array.
     */

    int * idxsrcrank = malloc(znum_nonzeros * sizeof(int));
    err = !idxsrcrank ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(apowneridx); free(apownerrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t i = 0; i < znum_nonzeros; i++) idxsrcrank[i] = -1;
    int * idxsrclocalidx = malloc(znum_nonzeros * sizeof(int));
    err = !idxsrclocalidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxsrcrank);
        free(apowneridx); free(apownerrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t i = 0; i < znum_nonzeros; i++) idxsrclocalidx[i] = -1;

    err = assumedpartition_read(
        size, apsize, apownerrank, apowneridx,
        znum_nonzeros, zidx,
        idxsrcrank, idxsrclocalidx, comm, disterr);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxsrclocalidx); free(idxsrcrank);
        free(apowneridx); free(apownerrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(apowneridx); free(apownerrank);

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "idxsrcrank=[");
            for (int64_t i = 0; i < znum_nonzeros; i++)
                fprintf(stderr, " %d", idxsrcrank[i]);
            fprintf(stderr, "], idxsrclocalidx=[");
            for (int64_t i = 0; i < znum_nonzeros; i++)
                fprintf(stderr, " %d", idxsrclocalidx[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 2: On each process, sort the offsets to the output vector
     * elements according to the ranks of processes that own the
     * corresponding elements of the input vector.
     *
     * The global offsets of the input vector elements needed by the
     * current process are now ready to send to each process that owns
     * those elements. Now, create a list of processes that the
     * current process will receive input vector elements from, and
     * count how many elements to request from each process.
     */

    int64_t * zperm = malloc(znum_nonzeros * sizeof(int64_t));
    err = !zperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxsrclocalidx); free(idxsrcrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    errno = radix_sort_int(znum_nonzeros, idxsrcrank, zperm);
    err = errno ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(zperm); free(idxsrclocalidx); free(idxsrcrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t idxsrcrankstart = 0;
    while (idxsrcrank[idxsrcrankstart] < 0) idxsrcrankstart++;
    int nrecvranks = idxsrcrankstart < znum_nonzeros ? 1 : 0;
    for (int64_t i = idxsrcrankstart+1; i < znum_nonzeros; i++) {
        if (idxsrcrank[i-1] != idxsrcrank[i])
            nrecvranks++;
    }
    int * recvranks = malloc(nrecvranks * sizeof(int));
    err = !recvranks ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(zperm); free(idxsrclocalidx); free(idxsrcrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * recvcounts = malloc(nrecvranks * sizeof(int));
    err = !recvcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvranks);
        free(zperm); free(idxsrclocalidx); free(idxsrcrank);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    if (nrecvranks > 0) {
        recvranks[0] = idxsrcrank[idxsrcrankstart];
        recvcounts[0] = 1;
    }
    for (int64_t i = idxsrcrankstart+1, p = 0; i < znum_nonzeros; i++) {
        if (idxsrcrank[i-1] != idxsrcrank[i]) {
            recvranks[++p] = idxsrcrank[i];
            recvcounts[p] = 0;
        }
        recvcounts[p]++;
    }
    free(idxsrcrank);

    int * rdispls = malloc((nrecvranks+1) * sizeof(int));
    err = !rdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvcounts); free(recvranks);
        free(zperm); free(idxsrclocalidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    rdispls[0] = 0;
    for (int p = 0; p < nrecvranks; p++)
        rdispls[p+1] = rdispls[p] + recvcounts[p];
    int idxsendcount = rdispls[nrecvranks];
    int64_t * idxsendbuf = malloc(idxsendcount * sizeof(int64_t));
    err = !idxsendbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm); free(idxsrclocalidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t i = 0; i < znum_nonzeros; i++) {
        if (zperm[i] >= idxsrcrankstart)
            idxsendbuf[zperm[i]-idxsrcrankstart] = idxsrclocalidx[i];
    }
    free(idxsrclocalidx);

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "idxsrcrankstart=%d, zperm=[", idxsrcrankstart);
            for (int64_t i = 0; i < znum_nonzeros; i++)
                fprintf(stderr, " %d", zperm[i]);
            fprintf(stderr, "], idxsendbuf=[");
            for (int64_t i = 0; i < idxsendcount; i++)
                fprintf(stderr, " %"PRId64, idxsendbuf[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 3: Count the number of processes requesting data from the
     * current process. This is achieved using a single counter on
     * each process and one-sided communication, so that every process
     * can atomically update remote counters on other processes, if it
     * needs data from them.
     */

    int nsendranks = 0;
    MPI_Win window;
    disterr->mpierrcode = MPI_Win_create(
        &nsendranks, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &window);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Win_fence(0, window);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    for (int p = 0; p < nrecvranks && !err; p++) {
        int one = 1;
        disterr->mpierrcode = MPI_Accumulate(
            &one, 1, MPI_INT, recvranks[p], 0, 1, MPI_INT, MPI_SUM, window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    }
    if (err) {
        MPI_Win_fence(0, window);
    } else {
        disterr->mpierrcode = MPI_Win_fence(0, window);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    }
    MPI_Win_free(&window);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "nrecvranks=%d, ", nrecvranks);
            fprintf(stderr, "recvranks=[");
            for (int64_t i = 0; i < nrecvranks; i++)
                fprintf(stderr, " %d", recvranks[i]);
            fprintf(stderr, "], recvcounts=[");
            for (int64_t i = 0; i < nrecvranks; i++)
                fprintf(stderr, " %d", recvcounts[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 4: Post non-blocking, wildcard receives for every process
     * requesting data from the current process. Thereafter, the
     * current process sends the number of requested elements to each
     * process from which it needs data.
     */

    MPI_Request * idxrequests = malloc(nsendranks * sizeof(MPI_Request));
    err = !idxrequests ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendranks = malloc(nsendranks * sizeof(int));
    err = !sendranks ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sendcounts = malloc(nsendranks * sizeof(int));
    err = !sendcounts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        disterr->mpierrcode = MPI_Irecv(
            &sendcounts[p], 1, MPI_INT, MPI_ANY_SOURCE, 3, comm, &idxrequests[p]);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        disterr->mpierrcode = MPI_Send(
            &recvcounts[p], 1, MPI_INT, recvranks[p], 3, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        MPI_Status status;
        disterr->mpierrcode = MPI_Wait(&idxrequests[p], &status);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
        sendranks[p] = status.MPI_SOURCE;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * sdispls = malloc((nsendranks+1) * sizeof(int));
    err = !sdispls ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    sdispls[0] = 0;
    for (int p = 0; p < nsendranks; p++)
        sdispls[p+1] = sdispls[p] + sendcounts[p];

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "nsendranks=%d, ", nrecvranks);
            fprintf(stderr, "sendranks=[");
            for (int64_t i = 0; i < nsendranks; i++)
                fprintf(stderr, " %d", sendranks[i]);
            fprintf(stderr, "], sendcounts=[");
            for (int64_t i = 0; i < nsendranks; i++)
                fprintf(stderr, " %d", sendcounts[i]);
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /* TODO: at this point, we could decide to sort ‘sendranks’ and
     * ‘sendcounts’ by ranks. Is there any benefit to doing so? */

    /*
     * Step 5: The current process sends arrays containing the global
     * offsets of its requested input vector elements to each process
     * that owns one or more of those input vector elements.
     */

    /* TODO: Here we can avoid allocating ‘idxrecvbuf’ and just use
     * ‘sendbuf.idx’ directly. */

    int64_t * idxrecvbuf = malloc(sdispls[nsendranks] * sizeof(int64_t));
    err = !idxrecvbuf ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(sdispls); free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nsendranks; p++) {
        disterr->mpierrcode = MPI_Irecv(
            &idxrecvbuf[sdispls[p]], sendcounts[p], MPI_INT64_T,
            sendranks[p], 4, comm, &idxrequests[p]);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxrecvbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < nrecvranks; p++) {
        disterr->mpierrcode = MPI_Send(
            &idxsendbuf[rdispls[p]], recvcounts[p], MPI_INT64_T, recvranks[p], 4, comm);
        err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (err) break;
    }
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxrecvbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(idxrequests); free(idxsendbuf);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    MPI_Waitall(nsendranks, idxrequests, MPI_STATUSES_IGNORE);
    free(idxsendbuf);
    free(idxrequests);

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_INIT
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "idxrecvbuf=[");
            for (int q = 0; q < nsendranks; q++) {
                if (q > 0) fprintf(stderr, " |");
                for (int64_t i = sdispls[q]; i < sdispls[q+1]; i++)
                    fprintf(stderr, " %"PRId64, idxrecvbuf[i]);
            }
            fprintf(stderr, "]\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 6: Allocate buffers to use for sending and receiving data.
     */

    struct mtxvector sendbuf;
    err = mtxvector_alloc_packed(
        &sendbuf, x->xp.type, field, precision,
        size, sdispls[nsendranks], idxrecvbuf);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(idxrecvbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(idxrecvbuf);

    int64_t * recvbufidx = malloc(rdispls[nrecvranks] * sizeof(int64_t));
    err = !recvbufidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&sendbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t i = 0; i < znum_nonzeros; i++) {
        if (zperm[i] >= idxsrcrankstart)
            recvbufidx[zperm[i]-idxsrcrankstart] = i;
    }
    struct mtxvector recvbuf;
    err = mtxvector_alloc_packed(
        &recvbuf, z->xp.type, field, precision,
        size, rdispls[nrecvranks], recvbufidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(recvbufidx);
        mtxvector_free(&sendbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(recvbufidx);
    MPI_Request * req = malloc(nrecvranks * sizeof(MPI_Request));
    err = !req ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxvector_free(&recvbuf);
        mtxvector_free(&sendbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    usscga->z = z;
    usscga->x = x;
    usscga->impl = malloc(sizeof(struct mtxmpivector_usscga_impl));
    err = !usscga->impl ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(req);
        mtxvector_free(&recvbuf);
        mtxvector_free(&sendbuf);
        free(sdispls); free(sendcounts); free(sendranks);
        free(rdispls); free(recvcounts); free(recvranks);
        free(zperm);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    usscga->impl->comm = comm;
    usscga->impl->commsize = comm_size;
    usscga->impl->rank = rank;
    usscga->impl->field = field;
    usscga->impl->precision = precision;
    usscga->impl->zperm = zperm;
    usscga->impl->idxsrcrankstart = idxsrcrankstart;
    usscga->impl->nrecvranks = nrecvranks;
    usscga->impl->recvranks = recvranks;
    usscga->impl->recvcounts = recvcounts;
    usscga->impl->rdispls = rdispls;
    usscga->impl->nsendranks = nsendranks;
    usscga->impl->sendranks = sendranks;
    usscga->impl->sendcounts = sendcounts;
    usscga->impl->sdispls = sdispls;
    usscga->impl->sendbuf = sendbuf;
    usscga->impl->recvbuf = recvbuf;
    usscga->impl->req = req;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_usscga_free()’ frees resources associated with a
 * persistent, combined scatter-gather operation.
 */
void mtxmpivector_usscga_free(
    struct mtxmpivector_usscga * usscga)
{
    free(usscga->impl->req);
    mtxvector_free(&usscga->impl->recvbuf);
    mtxvector_free(&usscga->impl->sendbuf);
    free(usscga->impl->sdispls);
    free(usscga->impl->sendcounts);
    free(usscga->impl->sendranks);
    free(usscga->impl->rdispls);
    free(usscga->impl->recvcounts);
    free(usscga->impl->recvranks);
    free(usscga->impl->zperm);
    free(usscga->impl);
}

/**
 * ‘mtxmpivector_usscga_start()’ initiates a combined scatter-gather
 * operation from a distributed sparse vector ‘x’ in packed form into
 * another distributed sparse vector ‘z’ in packed form. Repeated
 * indices in the packed vector ‘x’ are not allowed, otherwise the
 * result is undefined. They are, however, allowed in the packed
 * vector ‘z’.
 *
 * The operation may not complete before
 * ‘mtxmpivector_usscga_wait()’ is called.
 */
int mtxmpivector_usscga_start(
    struct mtxmpivector_usscga * usscga,
    struct mtxdisterror * disterr)
{
    MPI_Comm comm = usscga->impl->comm;

    /*
     * Step 7: For each process to which the current process must send data,
     * fill a buffer with the required input vector elements.
     */

    struct mtxvector * sendbuf = &usscga->impl->sendbuf;
    const struct mtxmpivector * x = usscga->x;
    int err = mtxvector_usga(sendbuf, &x->xp);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

#ifdef MTXDEBUG_MTXVECTOR_DIST_USSCGA_START
    int comm_size = usscga->impl->commsize;
    int rank = usscga->impl->rank;
    int64_t sendbufsize;
    mtxvector_size(sendbuf, &sendbufsize);
    for (int p = 0; p < comm_size; p++) {
        if (rank == p) {
            fprintf(stderr, "sendbuf=(");
            mtxvector_fwrite(
                sendbuf, sendbufsize, NULL, mtxfile_coordinate, stderr, NULL, NULL);
            fprintf(stderr, ")\n");
        }
        MPI_Barrier(comm);
        sleep(1);
    }
#endif

    /*
     * Step 8a: Send the requested input vector elements to the
     * destination processes.
     */

    struct mtxvector * recvbuf = &usscga->impl->recvbuf;
    int nrecvranks = usscga->impl->nrecvranks;
    const int * recvranks = usscga->impl->recvranks;
    const int * recvcounts = usscga->impl->recvcounts;
    const int * rdispls = usscga->impl->rdispls;
    MPI_Request * req = usscga->impl->req;
    for (int p = 0; p < nrecvranks; p++) {
        err = mtxvector_irecv(
            recvbuf, rdispls[p], recvcounts[p], recvranks[p],
            5, comm, &req[p], &disterr->mpierrcode);
        if (err) break;
    }
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int nsendranks = usscga->impl->nsendranks;
    const int * sendranks = usscga->impl->sendranks;
    const int * sendcounts = usscga->impl->sendcounts;
    const int * sdispls = usscga->impl->sdispls;
    for (int p = 0; p < nsendranks; p++) {
        err = mtxvector_send(
            sendbuf, sdispls[p], sendcounts[p], sendranks[p], 5, comm,
            &disterr->mpierrcode);
        if (err) break;
    }
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpivector_usscga_wait()’ waits for a persistent, combined
 * scatter-gather operation to finish.
 */
int mtxmpivector_usscga_wait(
    struct mtxmpivector_usscga * usscga,
    struct mtxdisterror * disterr)
{
    struct mtxmpivector * z = usscga->z;
    int nrecvranks = usscga->impl->nrecvranks;
    MPI_Request * req = usscga->impl->req;
    struct mtxvector * recvbuf = &usscga->impl->recvbuf;

    /*
     * Step 8b: Wait for data to be received.
     */

    MPI_Waitall(nrecvranks, req, MPI_STATUSES_IGNORE);

    /*
     * Step 9: Scatter the received input vector elements to their
     * final destinations in the output vector array.
     */

    int err = mtxvector_ussc(&z->xp, recvbuf);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}
#endif
