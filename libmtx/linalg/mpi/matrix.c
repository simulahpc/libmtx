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
 * Data structures and routines for distributed matrices.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/field.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/util/merge.h>
#include <libmtx/util/sort.h>
#include <libmtx/linalg/mpi/matrix.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/mpi/vector.h>

#include <mpi.h>

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ‘mtxmpimatrix_field()’ gets the field of a matrix.
 */
int mtxmpimatrix_field(
    const struct mtxmpimatrix * A,
    enum mtxfield * field)
{
    return mtxmatrix_field(&A->Ap, field);
}

/**
 * ‘mtxmpimatrix_precision()’ gets the precision of a matrix.
 */
int mtxmpimatrix_precision(
    const struct mtxmpimatrix * A,
    enum mtxprecision * precision)
{
    return mtxmatrix_precision(&A->Ap, precision);
}

/**
 * ‘mtxmpimatrix_symmetry()’ gets the symmetry of a matrix.
 */
int mtxmpimatrix_symmetry(
    const struct mtxmpimatrix * A,
    enum mtxsymmetry * symmetry)
{
    return mtxmatrix_symmetry(&A->Ap, symmetry);
}

/*
 * Memory management
 */

/**
 * ‘mtxmpimatrix_free()’ frees storage allocated for a matrix.
 */
void mtxmpimatrix_free(
    struct mtxmpimatrix * x)
{
    free(x->colmap);
    free(x->rowmap);
    mtxmatrix_free(&x->Ap);
}

static int mtxmpimatrix_init_comm(
    struct mtxmpimatrix * x,
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

static int mtxmpimatrix_init_size(
    struct mtxmpimatrix * x,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    /* check that dimensions are the same on all processes */
    int64_t psize[4] = {-num_rows, num_rows, -num_columns, num_columns};
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, psize, 4, MPI_INT64_T, MPI_MIN, comm);
    int err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    if (psize[0] != -psize[1]) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (psize[2] != -psize[3]) return MTX_ERR_INCOMPATIBLE_SIZE;
    x->num_rows = num_rows;
    x->num_columns = num_columns;

    /* sum the number of nonzeros across all processes */
    x->num_nonzeros = num_nonzeros;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &x->num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

static int mtxmpimatrix_init_rowmap_global(
    struct mtxmpimatrix * A,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    int * localrowidx,
    struct mtxdisterror * disterr)
{
    /* compute the mapping from local to global matrix rows */
    int64_t * globalrowidx = malloc(num_nonzeros * sizeof(int64_t));
    int err = !globalrowidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t * rowperm = malloc(num_nonzeros * sizeof(int64_t));
    err = !rowperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(globalrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * rowdstidx = malloc(num_nonzeros * sizeof(int64_t));
    err = !rowdstidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rowperm); free(globalrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t k = 0; k < num_nonzeros; k++)
        globalrowidx[k] = *(const int64_t *)((const char *) rowidx+k*idxstride) - idxbase;
    int64_t rowmapsize;
    err = compact_unsorted_int64(
        &rowmapsize, NULL, num_nonzeros, globalrowidx, rowperm, rowdstidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rowdstidx); free(rowperm); free(globalrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    A->rowmapsize = rowmapsize <= INT_MAX ? rowmapsize : -1;
    A->rowmap = malloc(A->rowmapsize * sizeof(int64_t));
    err = !A->rowmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(rowdstidx); free(rowperm); free(globalrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = compact_sorted_int64(
        &rowmapsize, A->rowmap, num_nonzeros, globalrowidx, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->rowmap); free(rowdstidx); free(rowperm); free(globalrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(globalrowidx);
    for (int64_t k = 0; k < num_nonzeros; k++)
        localrowidx[k] = rowdstidx[rowperm[k]];
    free(rowdstidx); free(rowperm);
    return MTX_SUCCESS;
}

static int mtxmpimatrix_init_colmap_global(
    struct mtxmpimatrix * A,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * colidx,
    int * localcolidx,
    struct mtxdisterror * disterr)
{
    /* compute the mapping from local to global matrix columns */
    int64_t * globalcolidx = malloc(num_nonzeros * sizeof(int64_t));
    int err = !globalcolidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t * colperm = malloc(num_nonzeros * sizeof(int64_t));
    err = !colperm ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(globalcolidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * coldstidx = malloc(num_nonzeros * sizeof(int64_t));
    err = !coldstidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(colperm); free(globalcolidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int64_t k = 0; k < num_nonzeros; k++)
        globalcolidx[k] = *(const int64_t *)((const char *) colidx+k*idxstride) - idxbase;
    int64_t colmapsize;
    err = compact_unsorted_int64(
        &colmapsize, NULL, num_nonzeros, globalcolidx, colperm, coldstidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(coldstidx); free(colperm); free(globalcolidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    A->colmapsize = colmapsize <= INT_MAX ? colmapsize : -1;
    A->colmap = malloc(A->colmapsize * sizeof(int64_t));
    err = !A->colmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(coldstidx); free(colperm); free(globalcolidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = compact_sorted_int64(
        &colmapsize, A->colmap, num_nonzeros, globalcolidx, NULL);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(coldstidx); free(colperm); free(globalcolidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(globalcolidx);
    for (int64_t k = 0; k < num_nonzeros; k++)
        localcolidx[k] = coldstidx[colperm[k]];
    free(coldstidx); free(colperm);
    return MTX_SUCCESS;
}

static int mtxmpimatrix_init_maps_global(
    struct mtxmpimatrix * A,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int * localrowidx,
    int * localcolidx,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_init_rowmap_global(
        A, num_nonzeros, idxstride, idxbase, rowidx, localrowidx, disterr);
    if (err) return err;
    err = mtxmpimatrix_init_colmap_global(
        A, num_nonzeros, idxstride, idxbase, colidx, localcolidx, disterr);
    if (err) { free(A->rowmap); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_alloc_copy()’ allocates a copy of a matrix without
 * initialising the values.
 */
int mtxmpimatrix_alloc_copy(
    struct mtxmpimatrix * dst,
    const struct mtxmpimatrix * src,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_init_copy()’ allocates a copy of a matrix and also
 * copies the values.
 */
int mtxmpimatrix_init_copy(
    struct mtxmpimatrix * dst,
    const struct mtxmpimatrix * src,
    struct mtxdisterror * disterr);

/*
 * Initialise matrices from entrywise data in coordinate format with
 * local row and column offsets and explicit mappings from local to
 * global rows and columns.
 */

/**
 * ‘mtxmpimatrix_alloc_entries_local()’ allocates storage for a
 * matrix based on entrywise data in coordinate format.
 */
int mtxmpimatrix_alloc_entries_local(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int * rowidx,
    const int * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_init_comm(A, comm, disterr);
    if (err) return err;
    err = mtxmpimatrix_init_size(
        A, num_rows, num_columns, num_nonzeros, comm, disterr);
    if (err) return err;

    /* copy mappings from local to global matrix rows/columns */
    A->rowmapsize = rowmapsize;
    A->rowmap = malloc(rowmapsize * sizeof(int64_t));
    err = !A->rowmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    for (int i = 0; i < rowmapsize; i++) A->rowmap[i] = rowmap[i];
    A->colmapsize = colmapsize;
    A->colmap = malloc(colmapsize * sizeof(int64_t));
    err = !A->colmap ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int i = 0; i < colmapsize; i++) A->colmap[i] = colmap[i];

    /* allocate storage for the local matrix */
    err = mtxmatrix_alloc_entries(
        &A->Ap, type, field, precision, symmetry,
        A->rowmapsize, A->colmapsize, num_nonzeros,
        idxstride, idxbase, rowidx, colidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* sum the number of nonzeros across all processes */
    err = mtxmatrix_num_nonzeros(&A->Ap, &A->num_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &A->num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_local_real_single()’ allocates and
 * initialises a matrix from data in coordinate format with real,
 * single precision coefficients.
 */
int mtxmpimatrix_init_entries_local_real_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_local(
        A, type, mtx_field_real, mtx_single, symmetry,
        num_rows, num_columns, rowmapsize, rowmap, colmapsize, colmap,
        num_nonzeros, sizeof(*rowidx), 0, rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_real_single(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_local_real_double()’ allocates and
 * initialises a matrix from data in coordinate format with real,
 * double precision coefficients.
 */
int mtxmpimatrix_init_entries_local_real_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_init_entries_local_complex_single()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * single precision coefficients.
 */
int mtxmpimatrix_init_entries_local_complex_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_init_entries_local_complex_double()’ allocates and
 * initialises a matrix from data in coordinate format with complex,
 * double precision coefficients.
 */
int mtxmpimatrix_init_entries_local_complex_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_init_entries_local_integer_single()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * single precision coefficients.
 */
int mtxmpimatrix_init_entries_local_integer_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_init_entries_local_integer_double()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * double precision coefficients.
 */
int mtxmpimatrix_init_entries_local_integer_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_init_entries_local_pattern()’ allocates and
 * initialises a matrix from data in coordinate format with integer,
 * double precision coefficients.
 */
int mtxmpimatrix_init_entries_local_pattern(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int rowmapsize,
    const int64_t * rowmap,
    int colmapsize,
    const int64_t * colmap,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr);

/*
 * Initialise matrices from entrywise data in coordinate format with
 * global row and columns offsets.
 */

/**
 * ‘mtxmpimatrix_alloc_entries_global()’ allocates a distributed
 * matrix, where the local part of the matrix on each process is
 * stored as a matrix of the given type.
 */
int mtxmpimatrix_alloc_entries_global(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxfield field,
    enum mtxprecision precision,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_init_comm(A, comm, disterr);
    if (err) return err;
    err = mtxmpimatrix_init_size(
        A, num_rows, num_columns, num_nonzeros, comm, disterr);
    if (err) return err;

    /* compute mappings from local to global matrix rows/columns */
    int * localrowidx = malloc(num_nonzeros * sizeof(int));
    err = !localrowidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int * localcolidx = malloc(num_nonzeros * sizeof(int));
    err = !localcolidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxmpimatrix_init_maps_global(
        A, num_nonzeros, idxstride, idxbase, rowidx, colidx,
        localrowidx, localcolidx, disterr);
    if (err) { free(localcolidx); free(localrowidx); return err; }

    /* allocate storage for the local matrix */
    err = mtxmatrix_alloc_entries(
        &A->Ap, type, field, precision, symmetry,
        A->rowmapsize, A->colmapsize, num_nonzeros,
        sizeof(*localrowidx), 0, localrowidx, localcolidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        free(localcolidx); free(localrowidx);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(localcolidx); free(localrowidx);

    /* sum the number of nonzeros across all processes */
    err = mtxmatrix_num_nonzeros(&A->Ap, &A->num_nonzeros);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &A->num_nonzeros, 1, MPI_INT64_T, MPI_SUM, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(A->colmap); free(A->rowmap);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_real_single()’ allocates and
 * initialises a matrix with real, single precision coefficients.
 *
 * On each process, ‘idx’ and ‘data’ are arrays of length
 * ‘num_nonzeros’, containing the global offsets and values,
 * respectively, of the matrix elements stored on the process. Note
 * that ‘num_nonzeros’ may differ from one process to the next.
 */
int mtxmpimatrix_init_entries_global_real_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_real, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_real_single(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_real_double()’ allocates and
 * initialises a matrix with real, double precision coefficients.
 */
int mtxmpimatrix_init_entries_global_real_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_real, mtx_double, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_real_double(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_complex_single()’ allocates and
 * initialises a matrix with complex, single precision coefficients.
 */
int mtxmpimatrix_init_entries_global_complex_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_complex, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_complex_single(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_complex_double()’ allocates and
 * initialises a matrix with complex, double precision coefficients.
 */
int mtxmpimatrix_init_entries_global_complex_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_complex, mtx_double, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_complex_double(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_integer_single()’ allocates and
 * initialises a matrix with integer, single precision coefficients.
 */
int mtxmpimatrix_init_entries_global_integer_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_integer, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_integer_single(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_integer_double()’ allocates and
 * initialises a matrix with integer, double precision coefficients.
 */
int mtxmpimatrix_init_entries_global_integer_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_integer, mtx_double, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_integer_double(&A->Ap, num_nonzeros, sizeof(*data), data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_pattern()’ allocates and
 * initialises a binary pattern matrix, where every entry has a value
 * of one.
 */
int mtxmpimatrix_init_entries_global_pattern(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    return mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_pattern, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, sizeof(*rowidx), 0,
        rowidx, colidx, comm, disterr);
}

/*
 * Initialise matrices from strided, entrywise data in coordinate
 * format with global row and columns offsets.
 */

/**
 * ‘mtxmpimatrix_init_entries_global_strided_real_single()’
 * allocates and initialises a matrix with real, single precision
 * coefficients.
 */
int mtxmpimatrix_init_entries_global_strided_real_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const float * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_real, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_real_single(&A->Ap, num_nonzeros, datastride, data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_strided_real_double()’
 * allocates and initialises a matrix with real, double precision
 * coefficients.
 */
int mtxmpimatrix_init_entries_global_strided_real_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const double * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_real, mtx_double, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_real_double(&A->Ap, num_nonzeros, datastride, data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_strided_complex_single()’
 * allocates and initialises a matrix with complex, single precision
 * coefficients.
 */
int mtxmpimatrix_init_entries_global_strided_complex_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_complex, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_complex_single(&A->Ap, num_nonzeros, datastride, data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_strided_complex_double()’
 * allocates and initialises a matrix with complex, double precision
 * coefficients.
 */
int mtxmpimatrix_init_entries_global_strided_complex_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_complex, mtx_double, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_complex_double(&A->Ap, num_nonzeros, datastride, data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_strided_integer_single()’
 * allocates and initialises a matrix with integer, single precision
 * coefficients.
 */
int mtxmpimatrix_init_entries_global_strided_integer_single(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_integer, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_integer_single(&A->Ap, num_nonzeros, datastride, data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_strided_integer_double()’
 * allocates and initialises a matrix with integer, double precision
 * coefficients.
 */
int mtxmpimatrix_init_entries_global_strided_integer_double(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    int datastride,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_integer, mtx_double, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
    if (err) return err;
    err = mtxmatrix_set_integer_double(&A->Ap, num_nonzeros, datastride, data);
    if (err) { mtxmpimatrix_free(A); return err; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_init_entries_global_pattern()’ allocates and
 * initialises a binary pattern matrix, where every entry has a value
 * of one.
 */
int mtxmpimatrix_init_entries_global_strided_pattern(
    struct mtxmpimatrix * A,
    enum mtxmatrixtype type,
    enum mtxsymmetry symmetry,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int idxstride,
    int idxbase,
    const int64_t * rowidx,
    const int64_t * colidx,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    return mtxmpimatrix_alloc_entries_global(
        A, type, mtx_field_pattern, mtx_single, symmetry,
        num_rows, num_columns, num_nonzeros, idxstride, idxbase,
        rowidx, colidx, comm, disterr);
}

/*
 * Modifying values
 */

/**
 * ‘mtxmpimatrix_set_constant_real_single()’ sets every nonzero
 * entry of a matrix equal to a constant, single precision floating
 * point number.
 */
int mtxmpimatrix_set_constant_real_single(
    struct mtxmpimatrix * x,
    float a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_set_constant_real_double()’ sets every nonzero
 * entry of a matrix equal to a constant, double precision floating
 * point number.
 */
int mtxmpimatrix_set_constant_real_double(
    struct mtxmpimatrix * x,
    double a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_set_constant_complex_single()’ sets every nonzero
 * entry of a matrix equal to a constant, single precision floating
 * point complex number.
 */
int mtxmpimatrix_set_constant_complex_single(
    struct mtxmpimatrix * x,
    float a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_set_constant_complex_double()’ sets every nonzero
 * entry of a matrix equal to a constant, double precision floating
 * point complex number.
 */
int mtxmpimatrix_set_constant_complex_double(
    struct mtxmpimatrix * x,
    double a[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_set_constant_integer_single()’ sets every nonzero
 * entry of a matrix equal to a constant integer.
 */
int mtxmpimatrix_set_constant_integer_single(
    struct mtxmpimatrix * x,
    int32_t a,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_set_constant_integer_double()’ sets every nonzero
 * entry of a matrix equal to a constant integer.
 */
int mtxmpimatrix_set_constant_integer_double(
    struct mtxmpimatrix * x,
    int64_t a,
    struct mtxdisterror * disterr);

/*
 * row and column vectors
 */

/**
 * ‘mtxmpimatrix_alloc_row_vector()’ allocates a row vector for a
 * given matrix, where a row vector is a vector whose length equal to
 * a single row of the matrix.
 */
int mtxmpimatrix_alloc_row_vector(
    const struct mtxmpimatrix * A,
    struct mtxmpivector * x,
    enum mtxvectortype type,
    struct mtxdisterror * disterr)
{
    enum mtxfield field;
    int err = mtxmpimatrix_field(A, &field);
    if (err) return err;
    enum mtxprecision precision;
    err = mtxmpimatrix_precision(A, &precision);
    if (err) return err;
    return mtxmpivector_alloc(
        x, type, field, precision, A->num_columns,
        A->colmapsize, A->colmap, A->comm, disterr);
}

/**
 * ‘mtxmpimatrix_alloc_column_vector()’ allocates a column vector
 * for a given matrix, where a column vector is a vector whose length
 * equal to a single column of the matrix.
 */
int mtxmpimatrix_alloc_column_vector(
    const struct mtxmpimatrix * A,
    struct mtxmpivector * y,
    enum mtxvectortype type,
    struct mtxdisterror * disterr)
{
    enum mtxfield field;
    int err = mtxmpimatrix_field(A, &field);
    if (err) return err;
    enum mtxprecision precision;
    err = mtxmpimatrix_precision(A, &precision);
    if (err) return err;
    return mtxmpivector_alloc(
        y, type, field, precision, A->num_rows,
        A->rowmapsize, A->rowmap, A->comm, disterr);
}

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxmpimatrix_from_mtxfile()’ converts from a matrix in Matrix
 * Market format.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxmatrix’ on each process.
 */
int mtxmpimatrix_from_mtxfile(
    struct mtxmpimatrix * A,
    const struct mtxfile * mtxfile,
    enum mtxmatrixtype type,
    MPI_Comm comm,
    int root,
    struct mtxdisterror * disterr)
{
    int err = mtxmpimatrix_init_comm(A, comm, disterr);
    if (err) return err;
    int comm_size = A->comm_size;
    int rank = A->rank;

    /* broadcast the header of the Matrix Market file */
    struct mtxfileheader mtxheader;
    if (rank == root) mtxheader = mtxfile->header;
    err = mtxfileheader_bcast(&mtxheader, root, comm, disterr);
    if (err) return err;
    if (mtxheader.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    enum mtxfield field;
    err = mtxfilefield_to_mtxfield(&field, mtxheader.field);
    if (err) return err;
    enum mtxsymmetry symmetry;
    err = mtxfilesymmetry_to_mtxsymmetry(&symmetry, mtxheader.symmetry);
    if (err) return err;

    /* broadcast the size of the Matrix Market file */
    struct mtxfilesize mtxsize;
    if (rank == root) mtxsize = mtxfile->size;
    err = mtxfilesize_bcast(&mtxsize, root, comm, disterr);
    if (err) return err;

    /* broadcast the precision */
    enum mtxprecision precision;
    if (rank == root) precision = mtxfile->precision;
    disterr->mpierrcode = MPI_Bcast(&precision, 1, MPI_INT, root, comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    /* divide entries or nonzeros into equal-sized blocks */
    int64_t num_rows = mtxsize.num_rows;
    int64_t num_columns = mtxsize.num_columns;
    int64_t recvsize;
    if (mtxheader.format == mtxfile_array) {
        int64_t num_entries = mtxsize.num_rows*mtxsize.num_columns;
        recvsize = num_entries / comm_size
            + (rank < (num_entries % comm_size) ? 1 : 0);
    } else if (mtxheader.format == mtxfile_coordinate) {
        recvsize = mtxsize.num_nonzeros / comm_size
            + (rank < (mtxsize.num_nonzeros % comm_size) ? 1 : 0);
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    err = mtxmpimatrix_init_size(
        A, num_rows, num_columns, recvsize, comm, disterr);
    if (err) return err;

    union mtxfiledata recvdata;
    err = mtxfiledata_alloc(
        &recvdata, mtxheader.object, mtxheader.format,
        mtxheader.field, precision, recvsize);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int64_t sendoffset = 0;
    for (int p = 0; p < comm_size; p++) {
        /* extract matrix market data for the current process */
        if (rank == root && p != root) {
            /* send from the root process */
            int64_t sendsize = 0;
            if (mtxheader.format == mtxfile_array) {
                int64_t num_entries = mtxsize.num_rows*mtxsize.num_columns;
                sendsize = num_entries / comm_size
                    + (p < (num_entries % comm_size) ? 1 : 0);
            } else if (mtxheader.format == mtxfile_coordinate) {
                sendsize = mtxsize.num_nonzeros / comm_size
                    + (p < (mtxsize.num_nonzeros % comm_size) ? 1 : 0);
            } else { err = err ? err : MTX_ERR_INVALID_MTX_FORMAT; }
            err = mtxfiledata_send(
                &mtxfile->data, mtxheader.object, mtxheader.format,
                mtxheader.field, precision, sendsize, sendoffset,
                p, 0, comm, disterr);
            if (err) MPI_Abort(comm, EXIT_FAILURE);
            sendoffset += sendsize;
        } else if (rank != root && rank == p) {
            /* receive from the root process */
            err = mtxfiledata_recv(
                &recvdata, mtxheader.object, mtxheader.format,
                mtxheader.field, precision, recvsize, 0,
                root, 0, comm, disterr);
            if (err) MPI_Abort(comm, EXIT_FAILURE);
        } else if (rank == root && p == root) {
            err = mtxfiledata_copy(
                &recvdata, &mtxfile->data,
                mtxheader.object, mtxheader.format,
                mtxheader.field, precision,
                recvsize, 0, sendoffset);
            if (err) MPI_Abort(comm, EXIT_FAILURE);
            sendoffset += recvsize;
        }
    }

    int idxstride;
    int64_t * rowidx;
    int64_t * colidx;
    err = mtxfiledata_rowcolidxptr(
        &recvdata, mtxheader.object, mtxheader.format,
        mtxheader.field, precision,
        &idxstride, &rowidx, &colidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfiledata_free(
            &recvdata, mtxheader.object, mtxheader.format,
            mtxheader.field, precision);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* compute mappings from local to global matrix rows/columns */
    int64_t num_nonzeros = recvsize;
    int * localrowidx = malloc(num_nonzeros * sizeof(int));
    err = !localrowidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfiledata_free(
            &recvdata, mtxheader.object, mtxheader.format,
            mtxheader.field, precision);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * localcolidx = malloc(num_nonzeros * sizeof(int));
    err = !localcolidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(localrowidx);
        mtxfiledata_free(
            &recvdata, mtxheader.object, mtxheader.format,
            mtxheader.field, precision);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxmpimatrix_init_maps_global(
        A, num_nonzeros, idxstride, 1, rowidx, colidx,
        localrowidx, localcolidx, disterr);
    if (err) {
        free(localcolidx); free(localrowidx);
        mtxfiledata_free(
            &recvdata, mtxheader.object, mtxheader.format,
            mtxheader.field, precision);
        return err;
    }

    err = mtxmatrix_alloc_entries(
        &A->Ap, type, field, precision, symmetry,
        A->rowmapsize, A->colmapsize, num_nonzeros,
        sizeof(*localrowidx), 0, localrowidx, localcolidx);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(localcolidx); free(localrowidx);
        mtxfiledata_free(
            &recvdata, mtxheader.object, mtxheader.format,
            mtxheader.field, precision);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(localcolidx); free(localrowidx);

    if (mtxheader.format == mtxfile_coordinate) {
        if (mtxheader.field == mtxfile_real) {
            if (precision == mtx_single) {
                err = mtxmatrix_set_real_single(
                    &A->Ap, num_nonzeros,
                    sizeof(*recvdata.matrix_coordinate_real_single),
                    &recvdata.matrix_coordinate_real_single[0].a);
            } else if (precision == mtx_double) {
                err = mtxmatrix_set_real_double(
                    &A->Ap, num_nonzeros,
                    sizeof(*recvdata.matrix_coordinate_real_double),
                    &recvdata.matrix_coordinate_real_double[0].a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxheader.field == mtxfile_complex) {
            if (precision == mtx_single) {
                err = mtxmatrix_set_complex_single(
                    &A->Ap, num_nonzeros,
                    sizeof(*recvdata.matrix_coordinate_complex_single),
                    &recvdata.matrix_coordinate_complex_single[0].a);
            } else if (precision == mtx_double) {
                err = mtxmatrix_set_complex_double(
                    &A->Ap, num_nonzeros,
                    sizeof(*recvdata.matrix_coordinate_complex_double),
                    &recvdata.matrix_coordinate_complex_double[0].a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxheader.field == mtxfile_integer) {
            if (precision == mtx_single) {
                err = mtxmatrix_set_integer_single(
                    &A->Ap, num_nonzeros,
                    sizeof(*recvdata.matrix_coordinate_integer_single),
                    &recvdata.matrix_coordinate_integer_single[0].a);
            } else if (precision == mtx_double) {
                err = mtxmatrix_set_integer_double(
                    &A->Ap, num_nonzeros,
                    sizeof(*recvdata.matrix_coordinate_integer_double),
                    &recvdata.matrix_coordinate_integer_double[0].a);
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxheader.field == mtxfile_pattern) {
            /* nothing to be done */
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else if (mtxheader.format == mtxfile_array) {
        /* TODO: initialise values for array matrices */
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    if (mtxdisterror_allreduce(disterr, err)) {
        mtxfiledata_free(
            &recvdata, mtxheader.object, mtxheader.format,
            mtxheader.field, precision);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxfiledata_free(
        &recvdata, mtxheader.object, mtxheader.format,
        mtxheader.field, precision);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_to_mtxfile()’ converts to a matrix in Matrix Market
 * format.
 */
int mtxmpimatrix_to_mtxfile(
    struct mtxfile * mtxfile,
    const struct mtxmpimatrix * A,
    enum mtxfileformat mtxfmt,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    enum mtxfield field;
    err = mtxmpimatrix_field(A, &field);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxprecision precision;
    err = mtxmpimatrix_precision(A, &precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxsymmetry symmetry;
    err = mtxmpimatrix_symmetry(A, &symmetry);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    struct mtxfileheader mtxheader;
    mtxheader.object = mtxfile_matrix;
    mtxheader.format = mtxfmt;
    if (field == mtx_field_real) mtxheader.field = mtxfile_real;
    else if (field == mtx_field_complex) mtxheader.field = mtxfile_complex;
    else if (field == mtx_field_integer) mtxheader.field = mtxfile_integer;
    else if (field == mtx_field_pattern) mtxheader.field = mtxfile_pattern;
    else { return MTX_ERR_INVALID_FIELD; }
    mtxheader.symmetry = mtxfile_general;

    struct mtxfilesize mtxsize;
    mtxsize.num_rows = A->num_rows;
    mtxsize.num_columns = A->num_columns;
    if (mtxfmt == mtxfile_array) {
        mtxsize.num_nonzeros = -1;
    } else if (mtxfmt == mtxfile_coordinate) {
        mtxsize.num_nonzeros = A->num_nonzeros;
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    if (A->rank == root)
        err = mtxfile_alloc(mtxfile, &mtxheader, NULL, &mtxsize, precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;

    int64_t recvoffset = 0;
    for (int p = 0; p < A->comm_size; p++) {
        if (A->rank == root && p != root) {
            /* receive at the root process */
            struct mtxfile recvmtxfile;
            err = err ? err : mtxfile_recv(&recvmtxfile, p, 0, A->comm, disterr);
            int64_t num_nonzeros = 0;
            if (mtxfile->header.format == mtxfile_array) {
                num_nonzeros = recvmtxfile.size.num_rows*recvmtxfile.size.num_columns;
            } else if (mtxfile->header.format == mtxfile_coordinate) {
                num_nonzeros = recvmtxfile.size.num_nonzeros;
            } else { err = MTX_ERR_INVALID_MTX_FORMAT; }
            err = err ? err : mtxfiledata_copy(
                &mtxfile->data, &recvmtxfile.data,
                recvmtxfile.header.object, recvmtxfile.header.format,
                recvmtxfile.header.field, recvmtxfile.precision,
                num_nonzeros, recvoffset, 0);
            mtxfile_free(&recvmtxfile);
            recvoffset += num_nonzeros;
        } else if (A->rank != root && A->rank == p) {
            /* send to the root process */
            struct mtxfile sendmtxfile;
            err = mtxmatrix_to_mtxfile(
                &sendmtxfile, &A->Ap, A->num_rows, A->rowmap,
                A->num_columns, A->colmap, mtxfmt);
            err = err ? err : mtxfile_send(&sendmtxfile, root, 0, A->comm, disterr);
            mtxfile_free(&sendmtxfile);
        } else if (A->rank == root && p == root) {
            struct mtxfile localmtxfile;
            err = mtxmatrix_to_mtxfile(
                &localmtxfile, &A->Ap, A->num_rows, A->rowmap,
                A->num_columns, A->colmap, mtxfmt);
            int64_t num_nonzeros = 0;
            if (mtxfile->header.format == mtxfile_array) {
                num_nonzeros = localmtxfile.size.num_rows*localmtxfile.size.num_columns;
            } else if (mtxfile->header.format == mtxfile_coordinate) {
                num_nonzeros = localmtxfile.size.num_nonzeros;
            } else { err = MTX_ERR_INVALID_MTX_FORMAT; }
            err = err ? err : mtxfiledata_copy(
                &mtxfile->data, &localmtxfile.data,
                localmtxfile.header.object, localmtxfile.header.format,
                localmtxfile.header.field, localmtxfile.precision,
                num_nonzeros, recvoffset, 0);
            mtxfile_free(&localmtxfile);
            recvoffset += num_nonzeros;
        }
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_from_mtxdistfile()’ converts from a matrix in
 * Matrix Market format that is distributed among multiple processes.
 *
 * The ‘type’ argument may be used to specify a desired storage format
 * or implementation for the underlying ‘mtxmatrix’ on each process.
 */
int mtxmpimatrix_from_mtxdistfile(
    struct mtxmpimatrix * A,
    const struct mtxdistfile * mtxdistfile,
    enum mtxmatrixtype type,
    MPI_Comm comm,
    struct mtxdisterror * disterr)
{
    if (mtxdistfile->header.object != mtxfile_matrix)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    enum mtxsymmetry symmetry;
    int err = mtxfilesymmetry_to_mtxsymmetry(
        &symmetry, mtxdistfile->header.symmetry);
    if (err) return err;
    int64_t num_rows = mtxdistfile->size.num_rows;
    int64_t num_columns = mtxdistfile->size.num_columns;
    int64_t num_nonzeros = mtxdistfile->localdatasize;

    if (mtxdistfile->header.format == mtxfile_array) {
        return MTX_ERR_INVALID_MTX_FORMAT;
        /* const int64_t * idx = mtxdistfile->idx; */
        /* if (mtxdistfile->header.field == mtxfile_real) { */
        /*     if (mtxdistfile->precision == mtx_single) { */
        /*         const float * data = mtxdistfile->data.array_real_single; */
        /*         err = mtxmpimatrix_init_real_single( */
        /*             A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, data, comm, disterr); */
        /*         if (err) { return err; } */
        /*     } else if (mtxdistfile->precision == mtx_double) { */
        /*         const double * data = mtxdistfile->data.array_real_double; */
        /*         err = mtxmpimatrix_init_real_double( */
        /*             A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, data, comm, disterr); */
        /*         if (err) { return err; } */
        /*     } else { return MTX_ERR_INVALID_PRECISION; } */
        /* } else if (mtxdistfile->header.field == mtxfile_complex) { */
        /*     if (mtxdistfile->precision == mtx_single) { */
        /*         const float (* data)[2] = mtxdistfile->data.array_complex_single; */
        /*         err = mtxmpimatrix_init_complex_single( */
        /*             A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, data, comm, disterr); */
        /*         if (err) { return err; } */
        /*     } else if (mtxdistfile->precision == mtx_double) { */
        /*         const double (* data)[2] = mtxdistfile->data.array_complex_double; */
        /*         err = mtxmpimatrix_init_complex_double( */
        /*             A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, data, comm, disterr); */
        /*         if (err) { return err; } */
        /*     } else { return MTX_ERR_INVALID_PRECISION; } */
        /* } else if (mtxdistfile->header.field == mtxfile_integer) { */
        /*     if (mtxdistfile->precision == mtx_single) { */
        /*         const int32_t * data = mtxdistfile->data.array_integer_single; */
        /*         err = mtxmpimatrix_init_integer_single( */
        /*             A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, data, comm, disterr); */
        /*         if (err) { return err; } */
        /*     } else if (mtxdistfile->precision == mtx_double) { */
        /*         const int64_t * data = mtxdistfile->data.array_integer_double; */
        /*         err = mtxmpimatrix_init_integer_double( */
        /*             A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, data, comm, disterr); */
        /*         if (err) { return err; } */
        /*     } else { return MTX_ERR_INVALID_PRECISION; } */
        /* } else if (mtxdistfile->header.field == mtxfile_pattern) { */
        /*     err = mtxmpimatrix_init_pattern( */
        /*         A, type, symmetry, num_rows, num_columns, num_nonzeros, idx, comm, disterr); */
        /*     if (err) { return err; } */
        /* } else { return MTX_ERR_INVALID_MTX_FIELD; } */
    } else if (mtxdistfile->header.format == mtxfile_coordinate) {
        if (mtxdistfile->header.field == mtxfile_real) {
            if (mtxdistfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_real_single * data =
                    mtxdistfile->data.matrix_coordinate_real_single;
                err = mtxmpimatrix_init_entries_global_strided_real_single(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else if (mtxdistfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_real_double * data =
                    mtxdistfile->data.matrix_coordinate_real_double;
                err = mtxmpimatrix_init_entries_global_strided_real_double(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_complex) {
            if (mtxdistfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_complex_single * data =
                    mtxdistfile->data.matrix_coordinate_complex_single;
                err = mtxmpimatrix_init_entries_global_strided_complex_single(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else if (mtxdistfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_complex_double * data =
                    mtxdistfile->data.matrix_coordinate_complex_double;
                err = mtxmpimatrix_init_entries_global_strided_complex_double(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_integer) {
            if (mtxdistfile->precision == mtx_single) {
                const struct mtxfile_matrix_coordinate_integer_single * data =
                    mtxdistfile->data.matrix_coordinate_integer_single;
                err = mtxmpimatrix_init_entries_global_strided_integer_single(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else if (mtxdistfile->precision == mtx_double) {
                const struct mtxfile_matrix_coordinate_integer_double * data =
                    mtxdistfile->data.matrix_coordinate_integer_double;
                err = mtxmpimatrix_init_entries_global_strided_integer_double(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j,
                    sizeof(*data), &data[0].a, comm, disterr);
                if (err) return err;
            } else { return MTX_ERR_INVALID_PRECISION; }
        } else if (mtxdistfile->header.field == mtxfile_pattern) {
            const struct mtxfile_matrix_coordinate_pattern * data =
                mtxdistfile->data.matrix_coordinate_pattern;
            err = mtxmpimatrix_init_entries_global_strided_pattern(
                    A, type, symmetry, num_rows, num_columns, num_nonzeros,
                    sizeof(*data), 1, &data[0].i, &data[0].j, comm, disterr);
            if (err) return err;
        } else { return MTX_ERR_INVALID_MTX_FIELD; }
    } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_to_mtxdistfile()’ converts to a matrix in Matrix
 * Market format that is distributed among multiple processes.
 */
int mtxmpimatrix_to_mtxdistfile(
    struct mtxdistfile * mtxdistfile,
    const struct mtxmpimatrix * x,
    enum mtxfileformat mtxfmt,
    struct mtxdisterror * disterr)
{
    /* MPI_Comm comm = x->comm; */
    /* int comm_size = x->comm_size; */
    /* int rank = x->rank; */
    /* enum mtxfield field; */
    /* int err = mtxmatrix_field(&x->Ap.x, &field); */
    /* if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE; */
    /* enum mtxprecision precision; */
    /* err = mtxmatrix_precision(&x->Ap.x, &precision); */
    /* if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE; */

    /* struct mtxfileheader mtxheader; */
    /* mtxheader.object = mtxfile_matrix; */
    /* mtxheader.format = mtxfmt; */
    /* if (field == mtx_field_real) mtxheader.field = mtxfile_real; */
    /* else if (field == mtx_field_complex) mtxheader.field = mtxfile_complex; */
    /* else if (field == mtx_field_integer) mtxheader.field = mtxfile_integer; */
    /* else if (field == mtx_field_pattern) mtxheader.field = mtxfile_pattern; */
    /* else { return MTX_ERR_INVALID_FIELD; } */
    /* mtxheader.symmetry = mtxfile_general; */

    /* struct mtxfilesize mtxsize; */
    /* mtxsize.num_rows = x->size; */
    /* mtxsize.num_columns = -1; */
    /* if (mtxfmt == mtxfile_array) { */
    /*     mtxsize.num_nonzeros = -1; */
    /* } else if (mtxfmt == mtxfile_coordinate) { */
    /*     mtxsize.num_nonzeros = x->num_nonzeros; */
    /* } else { return MTX_ERR_INVALID_MTX_FORMAT; } */

    /* err = mtxdistfile_alloc( */
    /*     mtxdistfile, &mtxheader, NULL, &mtxsize, precision, */
    /*     x->Ap.num_nonzeros, x->Ap.idx, x->comm, disterr); */
    /* if (err) return err; */

    /* struct mtxfile mtxfile; */
    /* err = mtxmatrix_to_mtxfile(&mtxfile, &x->Ap, mtxfmt); */
    /* if (mtxdisterror_allreduce(disterr, err)) { */
    /*     mtxdistfile_free(mtxdistfile); */
    /*     return MTX_ERR_MPI_COLLECTIVE; */
    /* } */

    /* err = mtxfiledata_copy( */
    /*     &mtxdistfile->data, &mtxfile.data, */
    /*     mtxfile.header.object, mtxfile.header.format, */
    /*     mtxfile.header.field, mtxfile.precision, */
    /*     x->Ap.num_nonzeros, 0, 0); */
    /* if (mtxdisterror_allreduce(disterr, err)) { */
    /*     mtxfile_free(&mtxfile); */
    /*     mtxdistfile_free(mtxdistfile); */
    /*     return MTX_ERR_MPI_COLLECTIVE; */
    /* } */
    /* mtxfile_free(&mtxfile); */
    return MTX_SUCCESS;
}

/*
 * I/O operations
 */

/**
 * ‘mtxmpimatrix_fwrite()’ writes a distributed matrix to a single
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
int mtxmpimatrix_fwrite(
    const struct mtxmpimatrix * x,
    enum mtxfileformat mtxfmt,
    FILE * f,
    const char * fmt,
    int64_t * bytes_written,
    int root,
    struct mtxdisterror * disterr)
{
    struct mtxdistfile dst;
    int err = mtxmpimatrix_to_mtxdistfile(&dst, x, mtxfmt, disterr);
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
 * ‘mtxmpimatrix_split()’ splits a matrix into multiple matrices
 * according to a given assignment of parts to each nonzero matrix
 * element.
 *
 * The partitioning of the nonzero matrix elements is specified by the
 * array ‘parts’. The length of the ‘parts’ array is given by ‘size’,
 * which must match the number of explicitly stored nonzero matrix
 * entries in ‘src’. Each entry in the ‘parts’ array is an integer in
 * the range ‘[0, num_parts)’ designating the part to which the
 * corresponding matrix nonzero belongs.
 *
 * The argument ‘dsts’ is an array of ‘num_parts’ pointers to objects
 * of type ‘struct mtxmpimatrix’. If successful, then ‘dsts[p]’
 * points to a matrix consisting of elements from ‘src’ that belong to
 * the ‘p’th part, as designated by the ‘parts’ array.
 *
 * The caller is responsible for calling ‘mtxmpimatrix_free()’ to
 * free storage allocated for each matrix in the ‘dsts’ array.
 */
int mtxmpimatrix_split(
    int num_parts,
    struct mtxmpimatrix ** dsts,
    const struct mtxmpimatrix * src,
    int64_t size,
    int * parts,
    int64_t * invperm,
    struct mtxdisterror * disterr)
{
    MPI_Comm comm = src->comm;
    enum mtxmatrixtype type = src->Ap.type;
    enum mtxfield field;
    int err = mtxmatrix_field(&src->Ap, &field);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxprecision precision;
    err = mtxmatrix_precision(&src->Ap, &precision);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    enum mtxsymmetry symmetry;
    err = mtxmatrix_symmetry(&src->Ap, &symmetry);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    int64_t num_rows = src->num_rows;
    int64_t num_columns = src->num_columns;

    struct mtxmatrix * matrices = malloc(num_parts * sizeof(struct mtxmatrix));
    err = !matrices ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    struct mtxmatrix ** matrixdsts = malloc(num_parts * sizeof(struct mtxmatrix *));
    err = !matrixdsts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) {
        free(matrices);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    for (int p = 0; p < num_parts; p++) matrixdsts[p] = &matrices[p];
    err = mtxmatrix_split(num_parts, matrixdsts, &src->Ap, size, parts/* , invperm */);
    if (mtxdisterror_allreduce(disterr, err)) {
        free(matrixdsts); free(matrices);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    free(matrixdsts);
    for (int p = 0; p < num_parts; p++) {
        int64_t dstsize;
        err = mtxmatrix_size(&matrices[p], &dstsize);
        if (mtxdisterror_allreduce(disterr, err)) {
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* obtain the local row and column indices of the matrix part */
        int * localrowidx = malloc(dstsize * sizeof(int));
        err = !localrowidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        int * localcolidx = malloc(dstsize * sizeof(int));
        err = !localcolidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(localrowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        err = mtxmatrix_rowcolidx(&matrices[p], dstsize, localrowidx, localcolidx);
        if (mtxdisterror_allreduce(disterr, err)) {
            free(localcolidx); free(localrowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }

        /* translate to global row and column indices */
        int64_t * globalrowidx = malloc(dstsize * sizeof(int64_t));
        err = !globalrowidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(localcolidx); free(localrowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        int64_t * globalcolidx = malloc(dstsize * sizeof(int64_t));
        err = !globalcolidx ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            free(globalrowidx);
            free(localcolidx); free(localrowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        for (int64_t k = 0; k < dstsize; k++) {
            globalrowidx[k] = src->rowmap[localrowidx[k]];
            globalcolidx[k] = src->colmap[localcolidx[k]];
        }
        free(localcolidx); free(localrowidx);

        /* allocate a new distributed matrix for the part, including
         * row and column maps */
        err = mtxmpimatrix_alloc_entries_global(
            dsts[p], type, field, precision, symmetry,
            num_rows, num_columns, dstsize, sizeof(*globalrowidx), 0,
            globalrowidx, globalcolidx, comm, disterr);
        if (err) {
            free(globalcolidx); free(globalrowidx);
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        free(globalcolidx); free(globalrowidx);

        /* copy data from the existing matrix */
        err = mtxmatrix_copy(&dsts[p]->Ap, &matrices[p]);
        if (mtxdisterror_allreduce(disterr, err)) {
            for (int q = p-1; q >= 0; q--) mtxmatrix_free(&dsts[q]->Ap);
            for (int q = p; q < num_parts; q++) mtxmatrix_free(&matrices[q]);
            free(matrices);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        mtxmatrix_free(&matrices[p]);
    }
    free(matrices);
    return MTX_SUCCESS;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘mtxmpimatrix_swap()’ swaps values of two matrices, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both matrices must also have the same number of
 * nonzero elements on that process.
 */
int mtxmpimatrix_swap(
    struct mtxmpimatrix * x,
    struct mtxmpimatrix * y,
    struct mtxdisterror * disterr)
{
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxmatrix_swap(&x->Ap, &y->Ap);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_copy()’ copies values of a matrix, ‘y = x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same total number of nonzero elements. On any
 * given process, both matrices must also have the same number of
 * nonzero elements on that process.
 */
int mtxmpimatrix_copy(
    struct mtxmpimatrix * y,
    const struct mtxmpimatrix * x,
    struct mtxdisterror * disterr)
{
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxmatrix_copy(&y->Ap, &x->Ap);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_sscal()’ scales a matrix by a single precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmpimatrix_sscal(
    float a,
    struct mtxmpimatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_sscal(a, &x->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_dscal()’ scales a matrix by a double precision
 * floating point scalar, ‘x = a*x’.
 */
int mtxmpimatrix_dscal(
    double a,
    struct mtxmpimatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_dscal(a, &x->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_cscal()’ scales a matrix by a complex, single
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmpimatrix_cscal(
    float a[2],
    struct mtxmpimatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_cscal(a, &x->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_zscal()’ scales a matrix by a complex, double
 * precision floating point scalar, ‘x = (a+b*i)*x’.
 */
int mtxmpimatrix_zscal(
    double a[2],
    struct mtxmpimatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_zscal(a, &x->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_saxpy()’ adds a matrix to another one multiplied by
 * a single precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the packed matrices are
 * allowed.
 */
int mtxmpimatrix_saxpy(
    float a,
    const struct mtxmpimatrix * x,
    struct mtxmpimatrix * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxmatrix_saxpy(a, &x->Ap, &y->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_daxpy()’ adds a matrix to another one multiplied by
 * a double precision floating point value, ‘y = a*x + y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the packed matrices are
 * allowed.
 */
int mtxmpimatrix_daxpy(
    double a,
    const struct mtxmpimatrix * x,
    struct mtxmpimatrix * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxmatrix_daxpy(a, &x->Ap, &y->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_saypx()’ multiplies a matrix by a single precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the packed matrices are
 * allowed.
 */
int mtxmpimatrix_saypx(
    float a,
    struct mtxmpimatrix * y,
    const struct mtxmpimatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxmatrix_saypx(a, &y->Ap, &x->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_daypx()’ multiplies a matrix by a double precision
 * floating point scalar and adds another matrix, ‘y = a*y + x’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. However, repeated indices in the packed matrices are
 * allowed.
 */
int mtxmpimatrix_daypx(
    double a,
    struct mtxmpimatrix * y,
    const struct mtxmpimatrix * x,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    int err = mtxmatrix_daypx(a, &y->Ap, &x->Ap, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_sdot()’ computes the Euclidean dot product of two
 * matrices in single precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_sdot(
    const struct mtxmpimatrix * x,
    const struct mtxmpimatrix * y,
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
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxmatrix_sdot(&x->Ap, &y->Ap, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_ddot()’ computes the Euclidean dot product of two
 * matrices in double precision floating point.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_ddot(
    const struct mtxmpimatrix * x,
    const struct mtxmpimatrix * y,
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
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxmatrix_ddot(&x->Ap, &y->Ap, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_cdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in single
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_cdotu(
    const struct mtxmpimatrix * x,
    const struct mtxmpimatrix * y,
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
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxmatrix_cdotu(&x->Ap, &y->Ap, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_zdotu()’ computes the product of the transpose of a
 * complex row matrix with another complex row matrix in double
 * precision floating point, ‘dot := x^T*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_zdotu(
    const struct mtxmpimatrix * x,
    const struct mtxmpimatrix * y,
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
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxmatrix_zdotu(&x->Ap, &y->Ap, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_cdotc()’ computes the Euclidean dot product of two
 * complex matrices in single precision floating point, ‘dot := x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_cdotc(
    const struct mtxmpimatrix * x,
    const struct mtxmpimatrix * y,
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
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxmatrix_cdotc(&x->Ap, &y->Ap, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_zdotc()’ computes the Euclidean dot product of two
 * complex matrices in double precision floating point, ‘dot := x^H*y’.
 *
 * The matrices ‘x’ and ‘y’ must have the same field, precision and
 * size, as well as the same number of nonzeros. On any given process,
 * both matrices must also have the same number of nonzero elements on
 * that process. The offsets of the nonzero entries are assumed to be
 * identical for both matrices, otherwise the results are
 * undefined. Moreover, repeated indices in the dist matrix are not
 * allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_zdotc(
    const struct mtxmpimatrix * x,
    const struct mtxmpimatrix * y,
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
    if (x->num_rows != y->num_rows) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_columns != y->num_columns) return MTX_ERR_INCOMPATIBLE_SIZE;
    if (x->num_nonzeros != y->num_nonzeros) return MTX_ERR_INCOMPATIBLE_SIZE;
    err = mtxmatrix_zdotc(&x->Ap, &y->Ap, dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_snrm2()’ computes the Euclidean norm of a matrix in
 * single precision floating point. Repeated indices in the dist
 * matrix are not allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_snrm2(
    const struct mtxmpimatrix * x,
    float * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    float dot[2] = {0.0f, 0.0f};
    int err = mtxmatrix_cdotc(&x->Ap, &x->Ap, &dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(dot[0]);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_dnrm2()’ computes the Euclidean norm of a matrix in
 * double precision floating point. Repeated indices in the dist
 * matrix are not allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_dnrm2(
    const struct mtxmpimatrix * x,
    double * nrm2,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    double dot[2] = {0.0, 0.0};
    int err = mtxmatrix_zdotc(&x->Ap, &x->Ap, &dot, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(dot[0]);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_sasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in single precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist matrix are not allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_sasum(
    const struct mtxmpimatrix * x,
    float * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_sasum(&x->Ap, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_dasum()’ computes the sum of absolute values
 * (1-norm) of a matrix in double precision floating point.  If the
 * matrix is complex-valued, then the sum of the absolute values of
 * the real and imaginary parts is computed. Repeated indices in the
 * dist matrix are not allowed, otherwise the result is undefined.
 */
int mtxmpimatrix_dasum(
    const struct mtxmpimatrix * x,
    double * asum,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err = mtxmatrix_dasum(&x->Ap, asum, num_flops);
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    disterr->mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, asum, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = disterr->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_iamax()’ finds the index of the first element
 * having the maximum absolute value.  If the matrix is
 * complex-valued, then the index points to the first element having
 * the maximum sum of the absolute values of the real and imaginary
 * parts. Repeated indices in the dist matrix are not allowed,
 * otherwise the result is undefined.
 */
int mtxmpimatrix_iamax(
    const struct mtxmpimatrix * x,
    int * iamax,
    struct mtxdisterror * disterr);

/*
 * Level 2 BLAS operations
 */

/**
 * ‘mtxmpimatrix_sgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another real scalar
 * ‘beta’ (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmpimatrix_sgemv(
    enum mtxtransposition trans,
    float alpha,
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    float beta,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    struct mtxmpimatrix_gemv gemv;
    enum mtxgemvoverlap overlap = mtxgemvoverlap_none;
    int err = mtxmpimatrix_gemv_init(&gemv, trans, A, x, y, overlap, disterr);
    if (err) { return err; }
    err = mtxmpimatrix_gemv_sgemv(&gemv, alpha, beta, disterr);
    if (err) { mtxmpimatrix_gemv_free(&gemv); return err; }
    err = mtxmpimatrix_gemv_wait(&gemv, num_flops, disterr);
    if (err) { mtxmpimatrix_gemv_free(&gemv); return err; }
    mtxmpimatrix_gemv_free(&gemv);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_dgemv()’ multiplies a matrix ‘A’ or its transpose
 * ‘A'’ by a real scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding
 * the result to another vector ‘y’ multiplied by another scalar real
 * ‘beta’ (‘β’).  That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmpimatrix_dgemv(
    enum mtxtransposition trans,
    double alpha,
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    double beta,
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    struct mtxmpimatrix_gemv gemv;
    enum mtxgemvoverlap overlap = mtxgemvoverlap_none;
    int err = mtxmpimatrix_gemv_init(&gemv, trans, A, x, y, overlap, disterr);
    if (err) { return err; }
    err = mtxmpimatrix_gemv_dgemv(&gemv, alpha, beta, disterr);
    if (err) { mtxmpimatrix_gemv_free(&gemv); return err; }
    err = mtxmpimatrix_gemv_wait(&gemv, num_flops, disterr);
    if (err) { mtxmpimatrix_gemv_free(&gemv); return err; }
    mtxmpimatrix_gemv_free(&gemv);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_cgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 */
int mtxmpimatrix_cgemv(
    enum mtxtransposition trans,
    float alpha[2],
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    float beta[2],
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_zgemv()’ multiplies a complex-valued matrix ‘A’,
 * its transpose ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another complex scalar ‘beta’
 * (‘β’).  That is, ‘y = α*A*x + β*y’, ‘y = α*A'*x + β*y’ or ‘y =
 * α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 */
int mtxmpimatrix_zgemv(
    enum mtxtransposition trans,
    double alpha[2],
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    double beta[2],
    struct mtxmpivector * y,
    int64_t * num_flops,
    struct mtxdisterror * disterr);

/*
 * persistent matrix-vector multiply operations, with optional
 * overlapping of computation and communication
 */

/**
 * ‘mtxmpimatrix_gemv_impl’ is an internal data structure for
 * persistent, matrix-vector multiply operations.
 */
struct mtxmpimatrix_gemv_impl
{
    struct mtxmpimatrix Aint;
    struct mtxmpimatrix Arowhalo;
    struct mtxmpimatrix Acolhalo;
    struct mtxmpimatrix Aext;
    struct mtxmpivector xint;
    struct mtxmpivector xhalo;
    bool yhalo_needed;
    struct mtxvector yint;
    struct mtxvector yhalo;
    struct mtxmpivector_usscga usscga_xint;
    struct mtxmpivector_usscga usscga_xhalo;

    struct mtxmpivector z;
    struct mtxmpivector w;
    int64_t num_flops;
    struct mtxmpivector_usscga usscga;
    enum mtxfield field;
    enum mtxprecision precision;
    float salpha, sbeta;
    double dalpha, dbeta;
};

/**
 * ‘mtxmpimatrix_gemv_rowparts()’ partitions the linear system
 * rowwise into interior and halo parts. The interior corresponds to
 * nonzero matrix rows where the current process also owns the
 * corresponding element of the destination vector, whereas the halo
 * corresponds to nonzero matrix rows where a remote process owns the
 * corresponding destination vector element.
 *
 * The array ‘dstrowparts’ must be of length ‘A->rowmapsize’.
 */
static int mtxmpimatrix_gemv_rowparts(
    enum mtxtransposition trans,
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    int * dstrowparts)
{
    int64_t num_nonzeros;
    int err = mtxvector_num_nonzeros(&x->xp, &num_nonzeros);
    if (err) return err;
    const int64_t * idx;
    err = mtxvector_idx(&x->xp, (int64_t **) &idx);
    if (err) return err;

    /* Make temporary copies of the row map (i.e., nonzero matrix rows
     * on the current process) and range map (i.e., the elements of
     * the destination vector, y, on the current process).  This is
     * needed, because the arrays must be sorted before computing the
     * intersection. */
    int64_t * rowmap = malloc(A->rowmapsize * sizeof(int64_t));
    if (!rowmap) return MTX_ERR_ERRNO;
    for (int64_t j = 0; j < A->rowmapsize; j++) rowmap[j] = A->rowmap[j];
    int64_t * rangemap = malloc(num_nonzeros * sizeof(int64_t));
    if (!rangemap) { free(rowmap); return MTX_ERR_ERRNO; }
    for (int64_t j = 0; j < num_nonzeros; j++) rangemap[j] = idx[j];
    int64_t * rowmapperm = malloc(A->rowmapsize * sizeof(int64_t));
    if (!rowmapperm) { free(rangemap); free(rowmap); return MTX_ERR_ERRNO; }
    int64_t * rangemapperm = malloc(num_nonzeros * sizeof(int64_t));
    if (!rangemapperm) { free(rowmapperm); free(rangemap); free(rowmap); return MTX_ERR_ERRNO; }

    /* compute the intersection of the row map and range map */
    int64_t introwmapsize = 0;
    err = setintersection_unsorted_unique_int64(
        &introwmapsize, NULL, A->rowmapsize, rowmap, rowmapperm, NULL,
        num_nonzeros, rangemap, rangemapperm, NULL);
    if (err) { free(rangemapperm); free(rowmapperm); free(rangemap); free(rowmap); return err; }
    int64_t * introwmap = malloc(introwmapsize * sizeof(int64_t));
    if (!introwmap) { free(rangemapperm); free(rowmapperm); free(rangemap); free(rowmap); return MTX_ERR_ERRNO; }
    int64_t * rowmapdstidx = malloc(A->rowmapsize * sizeof(int64_t));
    if (!rowmapdstidx) { free(introwmap); free(rangemapperm); free(rowmapperm); free(rangemap); free(rowmap); return MTX_ERR_ERRNO; }
    int64_t * rangemapdstidx = malloc(num_nonzeros * sizeof(int64_t));
    if (!rangemapdstidx) { free(rowmapdstidx); free(introwmap); free(rangemapperm); free(rowmapperm); free(rangemap); free(rowmap); return MTX_ERR_ERRNO; }
    err = setintersection_sorted_unique_int64(
        &introwmapsize, introwmap, A->rowmapsize, rowmap, rowmapdstidx,
        num_nonzeros, rangemap, rangemapdstidx);
    if (err) { free(rowmapdstidx); free(introwmap); free(rangemapperm); free(rowmapperm); free(rangemap); free(rowmap); return err; }
    free(rangemapdstidx); free(introwmap); free(rangemap); free(rowmap);

    /* partition matrix rows into interior and halo parts */
    for (int j = 0; j < A->rowmapsize; j++)
        dstrowparts[j] = rowmapdstidx[rowmapperm[j]] == -1 ? 1 : 0;
    free(rowmapdstidx); free(rangemapperm); free(rowmapperm);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_gemv_colparts()’ partitions the linear system
 * columnwise into interior and halo parts. The interior corresponds
 * to nonzero matrix columns where the current process also owns the
 * corresponding element of the source vector, whereas the halo
 * corresponds to nonzero matrix columns where a remote process owns
 * the corresponding source vector element.
 *
 * The array ‘dstcolparts’ must be of length ‘A->colmapsize’.
 */
static int mtxmpimatrix_gemv_colparts(
    enum mtxtransposition trans,
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    int * dstcolparts,
    int * dstxcolparts)
{
    int64_t num_nonzeros;
    int err = mtxvector_num_nonzeros(&x->xp, &num_nonzeros);
    if (err) return err;
    const int64_t * idx;
    err = mtxvector_idx(&x->xp, (int64_t **) &idx);
    if (err) return err;

    /* Make temporary copies of the column map (i.e., nonzero matrix
     * columns on the current process) and domain map (i.e., the
     * elements of the source vector, x, on the current process).
     * This is needed, because the arrays must be sorted before
     * computing the intersection. */
    int64_t * colmap = malloc(A->colmapsize * sizeof(int64_t));
    if (!colmap) return MTX_ERR_ERRNO;
    for (int64_t j = 0; j < A->colmapsize; j++) colmap[j] = A->colmap[j];
    int64_t * domainmap = malloc(num_nonzeros * sizeof(int64_t));
    if (!domainmap) { free(colmap); return MTX_ERR_ERRNO; }
    for (int64_t j = 0; j < num_nonzeros; j++) domainmap[j] = idx[j];
    int64_t * colmapperm = malloc(A->colmapsize * sizeof(int64_t));
    if (!colmapperm) { free(domainmap); free(colmap); return MTX_ERR_ERRNO; }
    int64_t * domainmapperm = malloc(num_nonzeros * sizeof(int64_t));
    if (!domainmapperm) {
        free(colmapperm);
        free(domainmap); free(colmap);
        return MTX_ERR_ERRNO;
    }

    /* compute the intersection of the column map and domain map */
    int64_t intcolmapsize = 0;
    err = setintersection_unsorted_unique_int64(
        &intcolmapsize, NULL, A->colmapsize, colmap, colmapperm, NULL,
        num_nonzeros, domainmap, domainmapperm, NULL);
    if (err) {
        free(domainmapperm); free(colmapperm);
        free(domainmap); free(colmap);
        return err;
    }
    int64_t * intcolmap = malloc(intcolmapsize * sizeof(int64_t));
    if (!intcolmap) {
        free(domainmapperm); free(colmapperm);
        free(domainmap); free(colmap);
        return MTX_ERR_ERRNO;
    }
    int64_t * colmapdstidx = malloc(A->colmapsize * sizeof(int64_t));
    if (!colmapdstidx) {
        free(intcolmap); free(domainmapperm); free(colmapperm);
        free(domainmap); free(colmap);
        return MTX_ERR_ERRNO;
    }
    int64_t * domainmapdstidx = malloc(num_nonzeros * sizeof(int64_t));
    if (!domainmapdstidx) {
        free(colmapdstidx); free(intcolmap);
        free(domainmapperm); free(colmapperm);
        free(domainmap); free(colmap);
        return MTX_ERR_ERRNO;
    }
    err = setintersection_sorted_unique_int64(
        &intcolmapsize, intcolmap, A->colmapsize, colmap, colmapdstidx,
        num_nonzeros, domainmap, domainmapdstidx);
    if (err) {
        free(colmapdstidx); free(intcolmap);
        free(domainmapperm); free(colmapperm);
        free(domainmap); free(colmap);
        return err;
    }
    free(intcolmap); free(domainmap); free(colmap);

    /* partition matrix columns into interior and halo parts */
    for (int j = 0; j < A->colmapsize; j++)
        dstcolparts[j] = colmapdstidx[colmapperm[j]] == -1 ? 1 : 0;
    free(colmapdstidx); free(colmapperm);

    /* partition vector elements into interior and halo parts */
    for (int j = 0; j < num_nonzeros; j++) {
        /* fprintf(stderr, "%s:%d: j=%d of %d, x->idx[j]=%d, domainmapperm[j]=%d, domainmapdstidx[j]=%d\n", __FILE__, __LINE__, j, num_nonzeros, x->idx[j], domainmapperm[j], domainmapdstidx[j]); */
        dstxcolparts[j] = domainmapdstidx[domainmapperm[j]] == -1 ? 1 : 0;
    }
    free(domainmapdstidx); free(domainmapperm);
    return MTX_SUCCESS;
}

/*
 * partition the local parts of a distributed matrix into four parts:
 * interior, exterior, row halo and column halo.
 */

/**
 * ‘mtxmpimatrix_gemv_partition()’ partitions the part of the
 * distributed linear system residing on the current process. The
 * linear system is partitioned in a 2D manner into interior, halo and
 * exterior parts. The interior corresponds to matrix nonzeros, where
 * the current process also owns the corresponding elements of the
 * source and destination vector. The row halo corresponds to matrix
 * nonzeros where a remote process owns the corresponding destination
 * vector element, whereas the column halo corresponds to matrix
 * nonzeros where a remote process owns the corresponding source
 * vector element. Finally, the exterior corresponds to matrix
 * nonzeros for which a remote process owns both the source and
 * destination vector elements.
 */
static int mtxmpimatrix_gemv_partition(
    enum mtxtransposition trans,
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    const struct mtxmpivector * y,
    struct mtxmpimatrix * Aint,
    struct mtxmpimatrix * Aext,
    struct mtxmpimatrix * Arowhalo,
    struct mtxmpimatrix * Acolhalo,
    struct mtxmpivector * xint,
    struct mtxmpivector * xhalo,
    bool * yhalo_needed,
    struct mtxvector * yint,
    struct mtxvector * yhalo,
    struct mtxdisterror * disterr)
{
    int64_t size;
    int err = mtxmatrix_size(&A->Ap, &size);
    if (err) return err;
    int64_t num_nonzeros;
    err = mtxvector_num_nonzeros(&x->xp, &num_nonzeros);
    if (err) return err;

    /* partition matrix rows into interior and halo parts */
    int * dstrowparts = malloc(A->rowmapsize * sizeof(int));
    if (!dstrowparts) return MTX_ERR_ERRNO;
    err = mtxmpimatrix_gemv_rowparts(trans, A, x, y, dstrowparts);
    if (err) { free(dstrowparts); return err; }

    /* partition matrix columns into interior and halo parts */
    int * dstcolparts = malloc(A->colmapsize * sizeof(int));
    if (!dstcolparts) { free(dstrowparts); return MTX_ERR_ERRNO; }
    int * dstxcolparts = malloc(num_nonzeros * sizeof(int));
    if (!dstxcolparts) { free(dstcolparts); free(dstrowparts); return MTX_ERR_ERRNO; }
    err = mtxmpimatrix_gemv_colparts(trans, A, x, y, dstcolparts, dstxcolparts);
    if (err) { free(dstxcolparts); free(dstcolparts); free(dstrowparts); return err; }

    /* partition matrix nonzeros accordingly */
    int * dstnzparts = malloc(size * sizeof(int));
    if (!dstnzparts) { free(dstcolparts); free(dstrowparts); return MTX_ERR_ERRNO; }
    int * ydstrowparts = malloc(A->rowmapsize * sizeof(int));
    if (!ydstrowparts) {
        free(dstnzparts); free(dstcolparts); free(dstrowparts);
        return MTX_ERR_ERRNO;
    }

    int64_t dstnzpartsizes[4] = {};
    int64_t dstrowpartsizes[2] = {};
    int64_t dstcolpartsizes[2] = {};

#if 0
    for (int p = 0; p < A->comm_size; p++) {
        if (A->rank == p) {
            fprintf(stderr, "%s:%d dstrowparts=(", __FILE__, __LINE__);
            for (int i = 0; i < A->rowmapsize; i++) fprintf(stderr, " %d", dstrowparts[i]);
            fprintf(stderr, ") ");
            fprintf(stderr, "dstcolparts=(");
            for (int i = 0; i < A->colmapsize; i++) fprintf(stderr, " %d", dstcolparts[i]);
            fprintf(stderr, ")\n");
        }
        sleep(1);
        MPI_Barrier(A->comm);
    }
#endif

    err = mtxmatrix_partition_2d(
        &A->Ap,
        mtx_custom_partition, 2, NULL, 0, dstrowparts,
        mtx_custom_partition, 2, NULL, 0, dstcolparts,
        dstnzparts, dstnzpartsizes,
        ydstrowparts, dstrowpartsizes,
        NULL, dstcolpartsizes);
    if (err) {
        free(dstxcolparts); free(ydstrowparts);
        free(dstnzparts); free(dstcolparts); free(dstrowparts);
        return err;
    }

#if 0
    for (int p = 0; p < A->comm_size; p++) {
        if (A->rank == p) {
            fprintf(stderr, "%s:%d dstrowparts=(", __FILE__, __LINE__);
            for (int i = 0; i < A->rowmapsize; i++) fprintf(stderr, " %d", dstrowparts[i]);
            fprintf(stderr, ") ");
            fprintf(stderr, "dstcolparts=(");
            for (int i = 0; i < A->colmapsize; i++) fprintf(stderr, " %d", dstcolparts[i]);
            fprintf(stderr, ") ");
            fprintf(stderr, "dstnzparts=(");
            for (int i = 0; i < size; i++) fprintf(stderr, " %d", dstnzparts[i]);
            fprintf(stderr, ")\n");
        }
        fprintf(stderr, "A: %d-by-%d (%d), "
                "Aint: %d-by-%d (%d), "
                "Acolhalo: %d-by-%d (%d), "
                "Arowhalo: %d-by-%d (%d), "
                "Aext: %d-by-%d (%d)\n",
                A->num_rows, A->num_columns, A->num_nonzeros,
                dstrowpartsizes[0], dstcolpartsizes[0], dstnzpartsizes[0],
                dstrowpartsizes[0], dstcolpartsizes[1], dstnzpartsizes[1],
                dstrowpartsizes[1], dstcolpartsizes[0], dstnzpartsizes[2],
                dstrowpartsizes[1], dstcolpartsizes[1], dstnzpartsizes[3]);
        sleep(1);
        MPI_Barrier(A->comm);
    }
#endif

    /* split the matrix into interior, exterior and halo parts */
    struct mtxmpimatrix * dsts[4] = {Aint, Acolhalo, Arowhalo, Aext};
    err = mtxmpimatrix_split(4, dsts, A, size, dstnzparts, NULL, disterr);
    if (err) {
        free(dstxcolparts); free(ydstrowparts);
        free(dstnzparts); free(dstcolparts); free(dstrowparts);
        return err;
    }
    free(dstnzparts);

    *yhalo_needed = dstrowpartsizes[1] > 0;
    if (*yhalo_needed) {
        /* split the destination vector into interior and halo parts */
        int64_t ynum_nonzeros;
        err = mtxvector_num_nonzeros(&y->xp, &ynum_nonzeros);
        if (err) {
            mtxmpimatrix_free(Aint); mtxmpimatrix_free(Aext);
            mtxmpimatrix_free(Acolhalo); mtxmpimatrix_free(Arowhalo);
            free(dstxcolparts); free(ydstrowparts);
            free(dstcolparts); free(dstrowparts);
            return err;
        }
        struct mtxvector * ydsts[2] = {yint, yhalo};
        err = mtxvector_split(
            2, ydsts, &y->xp, ynum_nonzeros, ydstrowparts, NULL);
        if (err) {
            mtxmpimatrix_free(Aint); mtxmpimatrix_free(Aext);
            mtxmpimatrix_free(Acolhalo); mtxmpimatrix_free(Arowhalo);
            free(dstxcolparts); free(ydstrowparts);
            free(dstcolparts); free(dstrowparts);
            return err;
        }
    }
    free(ydstrowparts);

    /* /\* split the source vector into interior and halo parts *\/ */
    /* struct mtxmpivector * xdsts[2] = {xint, xhalo}; */
    /* err = mtxmpivector_split( */
    /*     2, xdsts, x, num_nonzeros, dstxcolparts, NULL, disterr); */
    /* if (err) { */
    /*     free(dstxcolparts); free(dstcolparts); */
    /*     mtxvector_free(yint); mtxvector_free(yhalo); */
    /*     free(dstrowparts); */
    /*     return err; */
    /* } */

    /* allocate source vectors for the interior and halo parts */

    /* TODO: fix error handling */
    err = mtxmpimatrix_alloc_column_vector(Aint, xint, x->xp.type, disterr);
    err = mtxmpimatrix_alloc_column_vector(Acolhalo, xhalo, x->xp.type, disterr);
    free(dstxcolparts);
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_gemv_init()’ allocates data structures for a
 * persistent, matrix-vector multiply operation.
 *
 * This is used in cases where the matrix-vector multiply operation is
 * performed repeatedly, since the setup phase only needs to be
 * carried out once.
 */
int mtxmpimatrix_gemv_init(
    struct mtxmpimatrix_gemv * gemv,
    enum mtxtransposition trans,
    const struct mtxmpimatrix * A,
    const struct mtxmpivector * x,
    struct mtxmpivector * y,
    enum mtxgemvoverlap overlap,
    struct mtxdisterror * disterr)
{
    gemv->trans = trans;
    gemv->A = A;
    gemv->x = x;
    gemv->y = y;
    gemv->overlap = overlap;
    gemv->impl = malloc(sizeof(struct mtxmpimatrix_gemv_impl));
    int err = !gemv->impl ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
    struct mtxmpimatrix_gemv_impl * impl = gemv->impl;
    impl->num_flops = 0;

    if (overlap == mtxgemvoverlap_none) {
        /* no need to partition the matrix in this case */

        if (trans == mtx_notrans) {
            err = mtxmpimatrix_alloc_row_vector(
                A, &impl->z, x->xp.type, disterr);
        } else if (trans == mtx_trans) {
            err = mtxmpimatrix_alloc_column_vector(
                A, &impl->z, x->xp.type, disterr);
        } else { free(gemv->impl); return MTX_ERR_INVALID_TRANSPOSITION; }
        if (err) { free(gemv->impl); return err; }

        if (trans == mtx_notrans) {
            err = mtxmpimatrix_alloc_column_vector(
                A, &impl->w, y->xp.type, disterr);
        } else if (trans == mtx_trans) {
            err = mtxmpimatrix_alloc_row_vector(
                A, &impl->w, y->xp.type, disterr);
        } else {
            mtxmpivector_free(&impl->z);
            free(impl);
            return MTX_ERR_INVALID_TRANSPOSITION;
        }
        if (err) {
            mtxmpivector_free(&impl->z);
            free(impl);
            return err;
        }

        err = mtxmpivector_usscga_init(
            &gemv->impl->usscga, &gemv->impl->z, x, disterr);
        if (err) {
            mtxmpivector_free(&impl->w);
            mtxmpivector_free(&impl->z);
            free(impl);
            return err;
        }

    } else if (overlap == mtxgemvoverlap_irecv) {

        /* partition the local part of the linear system into
         * interior, exterior and halo parts. */
        err = mtxmpimatrix_gemv_partition(
            trans, A, x, y,
            &impl->Aint, &impl->Aext, &impl->Arowhalo, &impl->Acolhalo,
            &impl->xint, &impl->xhalo,
            &impl->yhalo_needed, &impl->yint, &impl->yhalo, disterr);
        if (err) return err;

        /* set up the communication for the source vector halo */
        err = mtxmpivector_usscga_init(&impl->usscga_xint, &impl->xint, x, disterr);
        if (err) {
            if (impl->yhalo_needed) { mtxvector_free(&impl->yint); mtxvector_free(&impl->yhalo); }
            mtxmpivector_free(&impl->xint); mtxmpivector_free(&impl->xhalo);
            mtxmpimatrix_free(&impl->Aint); mtxmpimatrix_free(&impl->Aext);
            mtxmpimatrix_free(&impl->Arowhalo); mtxmpimatrix_free(&impl->Acolhalo);
            free(impl);
            return err;
        }

        /* set up the communication for the source vector halo */
        err = mtxmpivector_usscga_init(&impl->usscga_xhalo, &impl->xhalo, x, disterr);
        if (err) {
            mtxmpivector_usscga_free(&impl->usscga_xint);
            if (impl->yhalo_needed) { mtxvector_free(&impl->yint); mtxvector_free(&impl->yhalo); }
            mtxmpivector_free(&impl->xint); mtxmpivector_free(&impl->xhalo);
            mtxmpimatrix_free(&impl->Aint); mtxmpimatrix_free(&impl->Aext);
            mtxmpimatrix_free(&impl->Arowhalo); mtxmpimatrix_free(&impl->Acolhalo);
            free(impl);
            return err;
        }

    } else { free(impl); return MTX_ERR_INVALID_GEMVOVERLAP; }

    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_gemv_free()’ frees resources associated with a
 * persistent, matrix-vector multiply operation.
 */
void mtxmpimatrix_gemv_free(
    struct mtxmpimatrix_gemv * gemv)
{
    if (gemv->overlap == mtxgemvoverlap_none) {
        mtxmpivector_usscga_free(&gemv->impl->usscga);
        mtxmpivector_free(&gemv->impl->w);
        mtxmpivector_free(&gemv->impl->z);
    } else if (gemv->overlap == mtxgemvoverlap_irecv) {
        mtxmpivector_usscga_free(&gemv->impl->usscga_xint);
        mtxmpivector_usscga_free(&gemv->impl->usscga_xhalo);
        if (gemv->impl->yhalo_needed) { mtxvector_free(&gemv->impl->yint); mtxvector_free(&gemv->impl->yhalo); }
        mtxmpivector_free(&gemv->impl->xint); mtxmpivector_free(&gemv->impl->xhalo);
        mtxmpimatrix_free(&gemv->impl->Aint); mtxmpimatrix_free(&gemv->impl->Aext);
        mtxmpimatrix_free(&gemv->impl->Arowhalo); mtxmpimatrix_free(&gemv->impl->Acolhalo);
    }
    free(gemv->impl);
}

/**
 * ‘mtxmpimatrix_gemv_sgemv()’ initiates a matrix-vector multiply
 * operation to multiply a matrix ‘A’ or its transpose ‘A'’ by a real
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another real scalar ‘beta’
 * (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmpimatrix_gemv_wait()’
 * is called.
 */
int mtxmpimatrix_gemv_sgemv(
    struct mtxmpimatrix_gemv * gemv,
    float alpha,
    float beta,
    struct mtxdisterror * disterr)
{
    struct mtxmpimatrix_gemv_impl * impl = gemv->impl;
    int err = mtxmpivector_usscga_start(&impl->usscga, disterr);
    if (err) return err;
    if (gemv->overlap == mtxgemvoverlap_none) {
        mtxmpivector_usscga_wait(&impl->usscga, disterr);
        if (err) return err;
        err = mtxmpivector_setzero(&impl->w, disterr);
        if (err) return err;
        err = mtxmatrix_sgemv(
            gemv->trans, alpha, &gemv->A->Ap,
            &impl->z.xp, 1.0f, &impl->w.xp, &impl->num_flops);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        if (gemv->trans == mtx_notrans) {
            err = mtxvector_saypx(
                beta, &gemv->y->xp, &impl->w.xp, &impl->num_flops);
            if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        } else if (gemv->trans == mtx_trans) {
            err = mtxvector_saypx(
                beta, &gemv->y->xp, &impl->w.xp, &impl->num_flops);
            if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }
    } else {
        /* TODO: allow overlapping computation with communication */
        return MTX_ERR_INVALID_GEMVOVERLAP;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_gemv_dgemv()’ initiates a matrix-vector multiply
 * operation to multiply a matrix ‘A’ or its transpose ‘A'’ by a real
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’ multiplied by another real scalar ‘beta’
 * (‘β’). That is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmpimatrix_gemv_wait()’
 * is called.
 */
int mtxmpimatrix_gemv_dgemv(
    struct mtxmpimatrix_gemv * gemv,
    double alpha,
    double beta,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxmpimatrix_gemv_impl * impl = gemv->impl;

    if (gemv->overlap == mtxgemvoverlap_none) {
        err = mtxmpivector_usscga_start(&impl->usscga, disterr);
        if (err) return err;
        mtxmpivector_usscga_wait(&impl->usscga, disterr);
        if (err) return err;
        err = mtxmpivector_setzero(&impl->w, disterr);
        if (err) return err;
        err = mtxmatrix_dgemv(
            gemv->trans, alpha, &gemv->A->Ap,
            &impl->z.xp, 1.0, &impl->w.xp, &impl->num_flops);
        if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        if (gemv->trans == mtx_notrans) {
            err = mtxvector_daypx(
                beta, &gemv->y->xp, &impl->w.xp, &impl->num_flops);
            if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        } else if (gemv->trans == mtx_trans) {
            err = mtxvector_daypx(
                beta, &gemv->y->xp, &impl->w.xp, &impl->num_flops);
            if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE;
        } else { return MTX_ERR_INVALID_TRANSPOSITION; }

    } else if (gemv->overlap == mtxgemvoverlap_irecv) {

        impl->field = mtx_field_real;
        impl->precision = mtx_double;
        impl->dalpha = alpha;
        impl->dbeta = beta;

        /* TODO: gather values from the source vector, x, to the
         * temporary vector, xint, which is used for multiplying the
         * interior part of the matrix */
        /* err = mtxvector_ussc(&impl->xint.xp, &gemv->x->xp); */
        /* if (err) return err; */

        err = mtxmpivector_usscga_start(&impl->usscga_xhalo, disterr);
        if (err) return err;
        err = mtxmpivector_usscga_start(&impl->usscga_xint, disterr);
        if (err) return err;
        err = mtxmpivector_usscga_wait(&impl->usscga_xint, disterr);
        if (err) return err;

        /* multiply the interior part */
        if (!impl->yhalo_needed) {
            err = mtxmatrix_dgemv(
                gemv->trans, alpha, &gemv->impl->Aint.Ap,
                &impl->xint.xp, 1.0, &gemv->y->xp, &impl->num_flops);
            if (err) return err;
        } else {
            err = mtxvector_setzero(&impl->yint);
            if (err) return err;
            err = mtxvector_setzero(&impl->yhalo);
            if (err) return err;

            /* TODO: support the case where the matrix rows and
             * destination vector elements are not owned by the same
             * process */

            return MTX_ERR_INVALID_GEMVOVERLAP;
        }

    } else { return MTX_ERR_INVALID_GEMVOVERLAP; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxmpimatrix_gemv_cgemv()’ initiates a matrix-vector multiply
 * operation to multiply a complex-values matrix ‘A’, its transpose
 * ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar ‘alpha’
 * (‘α’) and a vector ‘x’, before adding the result to another vector
 * ‘y’ multiplied by another complex ‘beta’ (‘β’). That is, ‘y = α*A*x
 * + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as single precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmpimatrix_gemv_wait()’
 * is called.
 */
int mtxmpimatrix_gemv_cgemv(
    struct mtxmpimatrix_gemv * gemv,
    float alpha[2],
    float beta[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_gemv_zgemv()’ initiates a matrix-vector multiply
 * operation to multiply a complex-values matrix ‘A’, its transpose
 * ‘A'’ or its conjugate transpose ‘Aᴴ’ by a complex scalar ‘alpha’
 * (‘α’) and a vector ‘x’, before adding the result to another vector
 * ‘y’ multiplied by another complex ‘beta’ (‘β’). That is, ‘y = α*A*x
 * + β*y’, ‘y = α*A'*x + β*y’ or ‘y = α*Aᴴ*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The operation may not complete before ‘mtxmpimatrix_gemv_wait()’
 * is called.
 */
int mtxmpimatrix_gemv_zgemv(
    struct mtxmpimatrix_gemv * gemv,
    double alpha[2],
    double beta[2],
    struct mtxdisterror * disterr);

/**
 * ‘mtxmpimatrix_gemv_wait()’ waits for a persistent, matrix-vector
 * multiply operation to finish.
 */
int mtxmpimatrix_gemv_wait(
    struct mtxmpimatrix_gemv * gemv,
    int64_t * num_flops,
    struct mtxdisterror * disterr)
{
    int err;
    struct mtxmpimatrix_gemv_impl * impl = gemv->impl;

    if (gemv->overlap == mtxgemvoverlap_none) {
        if (num_flops) *num_flops += gemv->impl->num_flops;
        gemv->impl->num_flops = 0;

    } else if (gemv->overlap == mtxgemvoverlap_irecv) {

        err = mtxmpivector_usscga_wait(&impl->usscga_xhalo, disterr);
        if (err) return err;

        if (impl->field == mtx_field_real) {
            if (impl->precision == mtx_single) {
            } else if (impl->precision == mtx_double) {
                if (!impl->yhalo_needed) {
                    err = mtxmatrix_dgemv(
                        gemv->trans, impl->dalpha, &gemv->impl->Acolhalo.Ap,
                        &impl->xhalo.xp, 1.0, &gemv->y->xp, &impl->num_flops);
                    if (mtxdisterror_allreduce(disterr, err))
                        return MTX_ERR_MPI_COLLECTIVE;
                } else {
                    /* TODO: support the case where the matrix rows and
                     * destination vector elements are not owned by the same
                     * process */
                    return MTX_ERR_INVALID_GEMVOVERLAP;
                }

                /* if (gemv->trans == mtx_notrans) { */
                /*     err = mtxvector_daypx( */
                /*         impl->dbeta, &gemv->y->xp, &impl->w.xp, &impl->num_flops); */
                /*     if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE; */
                /* } else if (gemv->trans == mtx_trans) { */
                /*     err = mtxvector_daypx( */
                /*         impl->dbeta, &gemv->y->xp, &impl->w.xp, &impl->num_flops); */
                /*     if (mtxdisterror_allreduce(disterr, err)) return MTX_ERR_MPI_COLLECTIVE; */
                /* } else { return MTX_ERR_INVALID_TRANSPOSITION; } */

            } else { return MTX_ERR_INVALID_PRECISION; }
        } else { return MTX_ERR_INVALID_FIELD; }

        if (num_flops) *num_flops += gemv->impl->num_flops;
        gemv->impl->num_flops = 0;

    } else {
        /* TODO: allow overlapping computation with communication */

        /* int err = mtxmpivector_usscga_wait(&gemv->impl->usscga, disterr); */
        /* if (err) return err; */

        return MTX_ERR_INVALID_GEMVOVERLAP;
    }

    return MTX_SUCCESS;
}
#endif
