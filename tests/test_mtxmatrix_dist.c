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
 * Last modified: 2022-05-04
 *
 * Unit tests for distributed matrices.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/mtxdistfile2.h>
#include <libmtx/matrix/dist.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char * program_invocation_short_name = "test_mtxmatrix_dist";

/**
 * ‘test_mtxmatrix_dist_from_mtxfile()’ tests converting Matrix
 *  Market files to matrices.
 */
int test_mtxmatrix_dist_from_mtxfile(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_single mtxdata[] = {
            {1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}, {3,1,7.0f}, {3,3,9.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(1.0f, a[0]);
            TEST_ASSERT_EQ(3.0f, a[1]);
            TEST_ASSERT_EQ(4.0f, a[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(7.0f, a[0]);
            TEST_ASSERT_EQ(9.0f, a[1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,1,7.0}, {3,3,9.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
            const double * a = Ap->a.data.real_double;
            TEST_ASSERT_EQ(1.0, a[0]);
            TEST_ASSERT_EQ(3.0, a[1]);
            TEST_ASSERT_EQ(4.0, a[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const double * a = Ap->a.data.real_double;
            TEST_ASSERT_EQ(7.0, a[0]);
            TEST_ASSERT_EQ(9.0, a[1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_single mtxdata[] = {
            {1,1,1.0f,-1.0f}, {1,3,3.0f,-3.0f}, {2,1,4.0f,-4.0f}, {3,1,7.0f,-7.0f}, {3,3,9.0f,-9.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
            const float (*a)[2] = Ap->a.data.complex_single;
            TEST_ASSERT_EQ(1.0f, a[0][0]); TEST_ASSERT_EQ(-1.0f, a[0][1]);
            TEST_ASSERT_EQ(3.0f, a[1][0]); TEST_ASSERT_EQ(-3.0f, a[1][1]);
            TEST_ASSERT_EQ(4.0f, a[2][0]); TEST_ASSERT_EQ(-4.0f, a[2][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const float (*a)[2] = Ap->a.data.complex_single;
            TEST_ASSERT_EQ(7.0f, a[0][0]); TEST_ASSERT_EQ(-7.0f, a[0][1]);
            TEST_ASSERT_EQ(9.0f, a[1][0]); TEST_ASSERT_EQ(-9.0f, a[1][1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_complex_double mtxdata[] = {
            {1,1,1.0,-1.0}, {1,3,3.0,-3.0}, {2,1,4.0,-4.0}, {3,1,7.0,-7.0}, {3,3,9.0,-9.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_double(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
            const double (*a)[2] = Ap->a.data.complex_double;
            TEST_ASSERT_EQ(1.0, a[0][0]); TEST_ASSERT_EQ(-1.0, a[0][1]);
            TEST_ASSERT_EQ(3.0, a[1][0]); TEST_ASSERT_EQ(-3.0, a[1][1]);
            TEST_ASSERT_EQ(4.0, a[2][0]); TEST_ASSERT_EQ(-4.0, a[2][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const double (*a)[2] = Ap->a.data.complex_double;
            TEST_ASSERT_EQ(7.0, a[0][0]); TEST_ASSERT_EQ(-7.0, a[0][1]);
            TEST_ASSERT_EQ(9.0, a[1][0]); TEST_ASSERT_EQ(-9.0, a[1][1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_single mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,1,7}, {3,3,9}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
            const int32_t * a = Ap->a.data.integer_single;
            TEST_ASSERT_EQ(1, a[0]);
            TEST_ASSERT_EQ(3, a[1]);
            TEST_ASSERT_EQ(4, a[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const int32_t * a = Ap->a.data.integer_single;
            TEST_ASSERT_EQ(7, a[0]);
            TEST_ASSERT_EQ(9, a[1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_integer_double mtxdata[] = {
            {1,1,1}, {1,3,3}, {2,1,4}, {3,1,7}, {3,3,9}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_integer_double(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
            const int64_t * a = Ap->a.data.integer_double;
            TEST_ASSERT_EQ(1, a[0]);
            TEST_ASSERT_EQ(3, a[1]);
            TEST_ASSERT_EQ(4, a[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const int64_t * a = Ap->a.data.integer_double;
            TEST_ASSERT_EQ(7, a[0]);
            TEST_ASSERT_EQ(9, a[1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        struct mtxfile_matrix_coordinate_pattern mtxdata[] = {
            {1,1}, {1,3}, {2,1}, {3,1}, {3,3}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_pattern(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coordinate, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(3, A.num_rows);
        TEST_ASSERT_EQ(3, A.num_columns);
        TEST_ASSERT_EQ(5, A.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A.rowmapsize);
            TEST_ASSERT_EQ(0, A.rowmap[0]);
            TEST_ASSERT_EQ(1, A.rowmap[1]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(0, Ap->colidx[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A.rowmapsize);
            TEST_ASSERT_EQ(2, A.rowmap[0]);
            TEST_ASSERT_EQ(2, A.colmapsize);
            TEST_ASSERT_EQ(0, A.colmap[0]);
            TEST_ASSERT_EQ(2, A.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coordinate, A.xp.type);
            const struct mtxmatrix_coordinate * Ap = &A.xp.storage.coordinate;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
        }
        mtxmatrix_dist_free(&A);
        mtxfile_free(&mtxfile);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}
#if 0
/**
 * ‘test_mtxmatrix_dist_to_mtxfile()’ tests converting matrices to
 * Matrix Market files.
 */
int test_mtxmatrix_dist_to_mtxfile(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ(5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
            const struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile.data.matrix_coordinate_real_single;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0f, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1.0f, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2.0f, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3.0f, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0f, 1.0f} : (double[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ(5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
            const struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile.data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1.0, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2.0, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3.0, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0 ? (float[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (float[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 3, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
            const struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile.data.matrix_coordinate_complex_single;
            TEST_ASSERT_EQ(2, data[0].i);
            TEST_ASSERT_EQ(1.0f, data[0].a[0]);
            TEST_ASSERT_EQ(1.0f, data[0].a[1]);
            TEST_ASSERT_EQ(4, data[1].i);
            TEST_ASSERT_EQ(1.0f, data[1].a[0]);
            TEST_ASSERT_EQ(2.0f, data[1].a[1]);
            TEST_ASSERT_EQ(6, data[2].i);
            TEST_ASSERT_EQ(3.0f, data[2].a[0]);
            TEST_ASSERT_EQ(0.0f, data[2].a[1]);
            mtxfile_free(&mtxfile);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0 ? (double[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (double[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 3, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
            const struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile.data.matrix_coordinate_complex_double;
            TEST_ASSERT_EQ(2, data[0].i);
            TEST_ASSERT_EQ(1.0, data[0].a[0]);
            TEST_ASSERT_EQ(1.0, data[0].a[1]);
            TEST_ASSERT_EQ(4, data[1].i);
            TEST_ASSERT_EQ(1.0, data[1].a[0]);
            TEST_ASSERT_EQ(2.0, data[1].a[1]);
            TEST_ASSERT_EQ(6, data[2].i);
            TEST_ASSERT_EQ(3.0, data[2].a[0]);
            TEST_ASSERT_EQ(0.0, data[2].a[1]);
            mtxfile_free(&mtxfile);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int32_t * xdata = rank == 0 ? (int32_t[2]) {1, 1} : (int32_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_integer_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
            const struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile.data.matrix_coordinate_integer_single;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int64_t * xdata = rank == 0 ? (int64_t[2]) {1, 1} : (int64_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_integer_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxmatrix_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
            const struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile.data.matrix_coordinate_integer_double;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_from_mtxdistfile2()’ tests converting
 *  distributed Matrix Market files to matrices.
 */
int test_mtxmatrix_dist_from_mtxdistfile2(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    {
        int num_rows = 9;
        const struct mtxfile_matrix_coordinate_real_single * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_single[2])
                {{1,1.0f}, {2,2.0f}})
            : ((const struct mtxfile_matrix_coordinate_real_single[1])
                {{4,4.0f}});
        int64_t num_nonzeros = 3;
        int64_t localdatasize = rank == 0 ? 2 : 1;

        struct mtxdistfile2 src;
        err = mtxdistfile2_init_matrix_coordinate_real_single(
            &src, num_rows, num_nonzeros, localdatasize, NULL,
            srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxmatrix_dist xdist;
        err = mtxmatrix_dist_from_mtxdistfile2(
            &xdist, &src, mtxmatrix_base, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(9, xdist.size);
        TEST_ASSERT_EQ(9, xdist.xp.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.num_nonzeros);
            TEST_ASSERT_EQ(0, xdist.xp.idx[0]);
            TEST_ASSERT_EQ(1, xdist.xp.idx[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.num_nonzeros);
            TEST_ASSERT_EQ(3, xdist.xp.idx[0]);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, xdist.xp.x.type);
        const struct mtxmatrix_base * xbase = &xdist.xp.x.storage.base;
        TEST_ASSERT_EQ(mtx_field_real, xbase->field);
        TEST_ASSERT_EQ(mtx_single, xbase->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xbase->size);
            TEST_ASSERT_EQ(1.0f, xbase->data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, xbase->data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xbase->size);
            TEST_ASSERT_EQ(4.0f, xbase->data.real_single[0]);
        }
        mtxmatrix_dist_free(&xdist);
        mtxdistfile2_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_to_mtxdistfile2()’ tests converting matrices to
 * distributed Matrix Market files.
 */
int test_mtxmatrix_dist_to_mtxdistfile2(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile2 mtxdistfile2;
        err = mtxmatrix_dist_to_mtxdistfile2(
            &mtxdistfile2, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(5, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile2.precision);
        TEST_ASSERT_EQ(5, mtxdistfile2.datasize);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxdistfile2.data.matrix_coordinate_real_single;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0f, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2.0f, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3.0f, data[2].a);
        }
        mtxdistfile2_free(&mtxdistfile2);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0f, 1.0f} : (double[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile2 mtxdistfile2;
        err = mtxmatrix_dist_to_mtxdistfile2(
            &mtxdistfile2, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(5, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile2.precision);
        TEST_ASSERT_EQ(5, mtxdistfile2.datasize);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxdistfile2.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3.0, data[2].a);
        }
        mtxdistfile2_free(&mtxdistfile2);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0 ? (float[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (float[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile2 mtxdistfile2;
        err = mtxmatrix_dist_to_mtxdistfile2(
            &mtxdistfile2, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ( 3, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile2.precision);
        TEST_ASSERT_EQ(3, mtxdistfile2.datasize);
        const struct mtxfile_matrix_coordinate_complex_single * data =
            mtxdistfile2.data.matrix_coordinate_complex_single;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(2, data[0].i);
            TEST_ASSERT_EQ(1.0f, data[0].a[0]);
            TEST_ASSERT_EQ(1.0f, data[0].a[1]);
            TEST_ASSERT_EQ(4, data[1].i);
            TEST_ASSERT_EQ(1.0f, data[1].a[0]);
            TEST_ASSERT_EQ(2.0f, data[1].a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(6, data[0].i);
            TEST_ASSERT_EQ(3.0f, data[0].a[0]);
            TEST_ASSERT_EQ(0.0f, data[0].a[1]);
        }
        mtxdistfile2_free(&mtxdistfile2);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0 ? (double[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (double[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile2 mtxdistfile2;
        err = mtxmatrix_dist_to_mtxdistfile2(
            &mtxdistfile2, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ( 3, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile2.precision);
        TEST_ASSERT_EQ(3, mtxdistfile2.datasize);
        const struct mtxfile_matrix_coordinate_complex_double * data =
            mtxdistfile2.data.matrix_coordinate_complex_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(2, data[0].i);
            TEST_ASSERT_EQ(1.0, data[0].a[0]);
            TEST_ASSERT_EQ(1.0, data[0].a[1]);
            TEST_ASSERT_EQ(4, data[1].i);
            TEST_ASSERT_EQ(1.0, data[1].a[0]);
            TEST_ASSERT_EQ(2.0, data[1].a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(6, data[0].i);
            TEST_ASSERT_EQ(3.0, data[0].a[0]);
            TEST_ASSERT_EQ(0.0, data[0].a[1]);
        }
        mtxdistfile2_free(&mtxdistfile2);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int32_t * xdata = rank == 0 ? (int32_t[2]) {1, 1} : (int32_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_integer_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile2 mtxdistfile2;
        err = mtxmatrix_dist_to_mtxdistfile2(
            &mtxdistfile2, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ( 5, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile2.precision);
        TEST_ASSERT_EQ(5, mtxdistfile2.datasize);
        const struct mtxfile_matrix_coordinate_integer_single * data =
            mtxdistfile2.data.matrix_coordinate_integer_single;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3, data[2].a);
        }
        mtxdistfile2_free(&mtxdistfile2);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int64_t * xdata = rank == 0 ? (int64_t[2]) {1, 1} : (int64_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_integer_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile2 mtxdistfile2;
        err = mtxmatrix_dist_to_mtxdistfile2(
            &mtxdistfile2, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ( 5, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile2.precision);
        TEST_ASSERT_EQ(5, mtxdistfile2.datasize);
        const struct mtxfile_matrix_coordinate_integer_double * data =
            mtxdistfile2.data.matrix_coordinate_integer_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3, data[2].a);
        }
        mtxdistfile2_free(&mtxdistfile2);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_swap()’ tests swapping values of two
 * matrices.
 */
int test_mtxmatrix_dist_swap(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_single(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.size);
        TEST_ASSERT_EQ_MSG(5, x.num_nonzeros, "x.num_nonzeros=%d", x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, x.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_real, x.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, x.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(1.0f, x.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ(0.0f, x.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(1.0f, x.xp.x.storage.base.data.real_single[2]);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(1.0f, y.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, y.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(3.0f, y.xp.x.storage.base.data.real_single[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_double(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.size);
        TEST_ASSERT_EQ_MSG(5, x.num_nonzeros, "x.num_nonzeros=%d", x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, x.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_real, x.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, x.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ(2.0, x.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(1.0, x.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ(0.0, x.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, x.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(1.0, x.xp.x.storage.base.data.real_double[2]);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(1.0, y.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, y.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(3.0, y.xp.x.storage.base.data.real_double[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,-1.0f}, {1.0f,-1.0f}})
            : ((float[3][2]) {{1.0f,-1.0f}, {2.0f,-2.0f}, {3.0f,-3.0f}});
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{2.0f,-2.0f}, {1.0f,-1.0f}})
            : ((float[3][2]) {{0.0f,0.0f}, {2.0f,-2.0f}, {1.0f,-1.0f}});
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_single(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.size);
        TEST_ASSERT_EQ_MSG(5, x.num_nonzeros, "x.num_nonzeros=%d", x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, x.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, x.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, x.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 2.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 0.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 2.0f, x.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.x.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.x.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.x.storage.base.data.complex_single[2][1]);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.x.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.x.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.x.storage.base.data.complex_single[2][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,-1.0f}, {1.0f,-1.0f}})
            : ((double[3][2]) {{1.0f,-1.0f}, {2.0f,-2.0f}, {3.0f,-3.0f}});
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{2.0f,-2.0f}, {1.0f,-1.0f}})
            : ((double[3][2]) {{0.0f,0.0f}, {2.0f,-2.0f}, {1.0f,-1.0f}});
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_double(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.size);
        TEST_ASSERT_EQ_MSG(5, x.num_nonzeros, "x.num_nonzeros=%d", x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, x.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, x.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, x.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 2.0f, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 0.0f, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 2.0f, x.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.x.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.x.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.x.storage.base.data.complex_double[2][1]);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.x.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.x.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.x.storage.base.data.complex_double[2][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_copy()’ tests copying values from one matrix
 * to another.
 */
int test_mtxmatrix_dist_copy(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_single(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(1.0f, y.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, y.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(3.0f, y.xp.x.storage.base.data.real_single[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_double(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(1.0, y.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, y.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(3.0, y.xp.x.storage.base.data.real_double[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,-1.0f}, {1.0f,-1.0f}})
            : ((float[3][2]) {{1.0f,-1.0f}, {2.0f,-2.0f}, {3.0f,-3.0f}});
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{2.0f,-2.0f}, {1.0f,-1.0f}})
            : ((float[3][2]) {{0.0f,0.0f}, {2.0f,-2.0f}, {1.0f,-1.0f}});
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_single(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.x.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.x.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.x.storage.base.data.complex_single[2][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,-1.0f}, {1.0f,-1.0f}})
            : ((double[3][2]) {{1.0f,-1.0f}, {2.0f,-2.0f}, {3.0f,-3.0f}});
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{2.0f,-2.0f}, {1.0f,-1.0f}})
            : ((double[3][2]) {{0.0f,0.0f}, {2.0f,-2.0f}, {1.0f,-1.0f}});
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_double(
            &y, mtxmatrix_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxmatrix_base, y.xp.x.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.x.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.x.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.x.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.x.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.x.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.x.storage.base.data.complex_double[2][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_scal()’ tests scaling matrices by a constant.
 */
int test_mtxmatrix_dist_scal(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.xp.x.storage.base.data.real_single[2]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.xp.x.storage.base.data.real_single[2]);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(6.0f, x.xp.x.storage.base.data.real_double[2]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(12.0f, x.xp.x.storage.base.data.real_double[2]);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(4.0f, x.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(0.0f, x.xp.x.storage.base.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.x.storage.base.data.complex_single[0][1]);
        }
        float as[2] = {2, 3};
        err = mtxmatrix_dist_cscal(as, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -4.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 20.0f, x.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(-16.0f, x.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 28.0f, x.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 24.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 36.0f, x.xp.x.storage.base.data.complex_single[0][1]);
        }
        double ad[2] = {2, 3};
        err = mtxmatrix_dist_zscal(ad, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -68.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(  28.0f, x.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(-116.0f, x.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(   8.0f, x.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( -60.0f, x.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 144.0f, x.xp.x.storage.base.data.complex_single[0][1]);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(2.0f, x.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(4.0f, x.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(0.0f, x.xp.x.storage.base.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 4.0f, x.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.x.storage.base.data.complex_double[0][1]);
        }
        float as[2] = {2, 3};
        err = mtxmatrix_dist_cscal(as, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -4.0, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 20.0, x.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(-16.0, x.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 28.0, x.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 24.0, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 36.0, x.xp.x.storage.base.data.complex_double[0][1]);
        }
        double ad[2] = {2, 3};
        err = mtxmatrix_dist_zscal(ad, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -68.0, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(  28.0, x.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(-116.0, x.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(   8.0, x.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( -60.0, x.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 144.0, x.xp.x.storage.base.data.complex_double[0][1]);
        }
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_axpy()’ tests multiplying a matrix by a
 * constant and adding the result to another matrix.
 */
int test_mtxmatrix_dist_axpy(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_single(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 3.0f, y.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 2.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 6.0f, y.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ( 7.0f, y.xp.x.storage.base.data.real_single[2]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 5.0f, y.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.xp.x.storage.base.data.real_single[2]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 9.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.xp.x.storage.base.data.real_single[2]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.xp.x.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.xp.x.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.xp.x.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.xp.x.storage.base.data.real_single[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_double(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 3.0, y.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 2.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 6.0, y.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ( 7.0, y.xp.x.storage.base.data.real_double[2]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 5.0, y.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.xp.x.storage.base.data.real_double[2]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 9.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.xp.x.storage.base.data.real_double[2]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.xp.x.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.xp.x.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.xp.x.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.xp.x.storage.base.data.real_double[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{2.0f,1.0f}, {0.0f,2.0f}})
            : ((float[1][2]) {{1.0f,0.0f}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_single(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(2.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(6.0f, y.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(0.0f, y.xp.x.storage.base.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 5.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 4.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(10.0f, y.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.xp.x.storage.base.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 9.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(22.0f, y.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.xp.x.storage.base.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.xp.x.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(19.0f, y.xp.x.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(46.0f, y.xp.x.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0f, y.xp.x.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.xp.x.storage.base.data.complex_single[0][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0,1.0}, {1.0,2.0}})
            : ((double[1][2]) {{3.0,0.0}});
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{2.0,1.0}, {0.0,2.0}})
            : ((double[1][2]) {{1.0,0.0}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_double(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(2.0, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(6.0, y.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(0.0, y.xp.x.storage.base.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 5.0, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 4.0, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(10.0, y.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.xp.x.storage.base.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 9.0, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(22.0, y.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.xp.x.storage.base.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.xp.x.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(19.0, y.xp.x.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(46.0, y.xp.x.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0, y.xp.x.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.xp.x.storage.base.data.complex_double[0][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_dot()’ tests computing the dot products of
 * pairs of matrices.
 */
int test_mtxmatrix_dist_dot(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        float * ydata = rank == 0 ? (float[2]) {3.0f, 2.0f} : (float[3]) {1.0f, 0.0f, 1.0f};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_single(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxmatrix_dist_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxmatrix_dist_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxmatrix_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxmatrix_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxmatrix_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxmatrix_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        double * ydata = rank == 0 ? (double[2]) {3.0, 2.0} : (double[3]) {1.0, 0.0, 1.0};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_real_double(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxmatrix_dist_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxmatrix_dist_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxmatrix_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxmatrix_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxmatrix_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxmatrix_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{3.0f,2.0f}, {1.0f,0.0f}})
            : ((float[1][2]) {{1.0f,0.0f}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_single(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxmatrix_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxmatrix_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxmatrix_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxmatrix_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0,1.0}, {1.0,2.0}})
            : ((double[1][2]) {{3.0,0.0}});
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{3.0,2.0}, {1.0,0.0}})
            : ((double[1][2]) {{1.0,0.0}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_complex_double(
            &y, mtxmatrix_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxmatrix_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxmatrix_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxmatrix_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxmatrix_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_nrm2()’ tests computing the Euclidean norm of
 * matrices.
 */
int test_mtxmatrix_dist_nrm2(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxmatrix_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxmatrix_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxmatrix_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxmatrix_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxmatrix_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxmatrix_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxmatrix_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxmatrix_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_asum()’ tests computing the sum of
 * absolute values of matrices.
 */
int test_mtxmatrix_dist_asum(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
    int root = 0;
    int comm_size;
    err = MPI_Comm_size(comm, &comm_size);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    err = MPI_Comm_rank(comm, &rank);
    if (err) {
        MPI_Error_string(err, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (comm_size != 2) TEST_FAIL_MSG("Expected exactly two MPI processes");
    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err) MPI_Abort(comm, EXIT_FAILURE);
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {-1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxmatrix_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxmatrix_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {-1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxmatrix_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxmatrix_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{-1.0f,-1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxmatrix_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxmatrix_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{-1.0f,-1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxmatrix_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxmatrix_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}
#endif
/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;

    /* 1. initialise MPI. */
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const int mpi_root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* 2. run test suite. */
    TEST_SUITE_BEGIN("Running tests for distributed matrices\n");
    TEST_RUN(test_mtxmatrix_dist_from_mtxfile);
#if 0
    TEST_RUN(test_mtxmatrix_dist_to_mtxfile);
    TEST_RUN(test_mtxmatrix_dist_from_mtxdistfile2);
    TEST_RUN(test_mtxmatrix_dist_to_mtxdistfile2);
    TEST_RUN(test_mtxmatrix_dist_swap);
    TEST_RUN(test_mtxmatrix_dist_copy);
    TEST_RUN(test_mtxmatrix_dist_scal);
    TEST_RUN(test_mtxmatrix_dist_axpy);
    TEST_RUN(test_mtxmatrix_dist_dot);
    TEST_RUN(test_mtxmatrix_dist_nrm2);
    TEST_RUN(test_mtxmatrix_dist_asum);
#endif
    TEST_SUITE_END();

    /* 3. clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
