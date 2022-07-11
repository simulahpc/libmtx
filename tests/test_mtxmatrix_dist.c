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
 * Unit tests for distributed matrices.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/matrix/dist.h>
#include <libmtx/vector/dist.h>

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
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            {1,1,{1.0f,-1.0f}}, {1,3,{3.0f,-3.0f}}, {2,1,{4.0f,-4.0f}}, {3,1,{7.0f,-7.0f}}, {3,3,{9.0f,-9.0f}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            {1,1,{1.0,-1.0}}, {1,3,{3.0,-3.0}}, {2,1,{4.0,-4.0}}, {3,1,{7.0,-7.0}}, {3,3,{9.0,-9.0}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_matrix_coordinate_complex_double(
            &mtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxfile(
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            &A, &mtxfile, mtxmatrix_coo, comm, root, &disterr);
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
        int num_rows = 3;
        int num_columns = 3;
        int rowmapsize = rank == 0 ? 2 : 1;
        const int64_t * rowmap = (rank == 0)
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int colmapsize = 3;
        const int64_t * colmap = (rank == 0)
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const int * rowidx = (rank == 0)
            ? ((const int[3]) {0, 0, 1})
            : ((const int[2]) {0, 0});
        const int * colidx = (rank == 0)
            ? ((const int[3]) {0, 1, 2})
            : ((const int[2]) {0, 2});
        const float * srcdata = (rank == 0)
            ? ((const float[3]) {1.0f, 2.0f, 6.0f})
            : ((const float[2]) {7.0f, 9.0f});
        int64_t num_nonzeros = (rank == 0) ? 3 : 2;
        struct mtxmatrix_dist src;
        err = mtxmatrix_dist_init_entries_local_real_single(
            &src, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns,
            rowmapsize, rowmap, colmapsize, colmap,
            num_nonzeros, rowidx, colidx, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxmatrix_dist_to_mtxfile(
            &dst, &src, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(mtx_single, dst.precision);
            TEST_ASSERT_EQ(3, dst.size.num_rows);
            TEST_ASSERT_EQ(3, dst.size.num_columns);
            TEST_ASSERT_EQ(5, dst.size.num_nonzeros);
            const struct mtxfile_matrix_coordinate_real_single * data =
                dst.data.matrix_coordinate_real_single;
            TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ(   1, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(2.0f, data[1].a);
            TEST_ASSERT_EQ(   2, data[2].i); TEST_ASSERT_EQ(3, data[2].j);
            TEST_ASSERT_EQ(6.0f, data[2].a);
            TEST_ASSERT_EQ(   3, data[3].i); TEST_ASSERT_EQ(1, data[3].j);
            TEST_ASSERT_EQ(7.0f, data[3].a);
            TEST_ASSERT_EQ(   3, data[4].i); TEST_ASSERT_EQ(3, data[4].j);
            TEST_ASSERT_EQ(9.0f, data[4].a);
            mtxfile_free(&dst);
        }
        mtxmatrix_dist_free(&src);
    }
    {
        int num_rows = 3;
        int num_columns = 3;
        const int64_t * rowidx = (rank == 0)
            ? ((const int64_t[3]) {0, 0, 1})
            : ((const int64_t[2]) {2, 2});
        const int64_t * colidx = (rank == 0)
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[2]) {0, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 6.0})
            : ((const double[2]) {7.0, 9.0});
        int64_t num_nonzeros = (rank == 0) ? 3 : 2;
        struct mtxmatrix_dist src;
        err = mtxmatrix_dist_init_entries_global_real_double(
            &src, mtxmatrix_csr, mtx_unsymmetric, num_rows, num_columns,
            num_nonzeros, rowidx, colidx, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxmatrix_dist_to_mtxfile(
            &dst, &src, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(3, dst.size.num_rows);
            TEST_ASSERT_EQ(3, dst.size.num_columns);
            TEST_ASSERT_EQ(5, dst.size.num_nonzeros);
            const struct mtxfile_matrix_coordinate_real_double * data =
                dst.data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(  1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(  1, data[1].i); TEST_ASSERT_EQ(  2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(  2, data[2].i); TEST_ASSERT_EQ(  3, data[2].j);
            TEST_ASSERT_EQ(6.0, data[2].a);
            TEST_ASSERT_EQ(  3, data[3].i); TEST_ASSERT_EQ(  1, data[3].j);
            TEST_ASSERT_EQ(7.0, data[3].a);
            TEST_ASSERT_EQ(  3, data[4].i); TEST_ASSERT_EQ(  3, data[4].j);
            TEST_ASSERT_EQ(9.0, data[4].a);
            mtxfile_free(&dst);
        }
        mtxmatrix_dist_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_from_mtxdistfile()’ tests converting
 *  distributed Matrix Market files to matrices.
 */
int test_mtxmatrix_dist_from_mtxdistfile(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
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
        const struct mtxfile_matrix_coordinate_real_single * srcdata = rank == 0
            ? ((const struct mtxfile_matrix_coordinate_real_single[3])
                {{1,1,1.0f}, {1,3,3.0f}, {2,1,4.0f}})
            : ((const struct mtxfile_matrix_coordinate_real_single[2])
                {{3,1,7.0f}, {3,3,9.0f}});
        int64_t num_nonzeros = 5;
        int64_t localdatasize = rank == 0 ? 3 : 2;
        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_single(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros,
            localdatasize, NULL, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxmatrix_dist A;
        err = mtxmatrix_dist_from_mtxdistfile(
            &A, &src, mtxmatrix_coo, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
            TEST_ASSERT_EQ(mtxmatrix_coo, A.Ap.type);
            const struct mtxmatrix_coo * Ap = &A.Ap.storage.coo;
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
        mtxdistfile_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}
#if 0
/**
 * ‘test_mtxmatrix_dist_to_mtxdistfile()’ tests converting matrices to
 * distributed Matrix Market files.
 */
int test_mtxmatrix_dist_to_mtxdistfile(void)
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
            &x, mtxmatrix_coo, size, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxmatrix_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxdistfile.data.matrix_coordinate_real_single;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0f, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2.0f, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3.0f, data[2].a);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0f, 1.0f} : (double[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_real_double(
            &x, mtxmatrix_coo, size, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxmatrix_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxdistfile.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3.0, data[2].a);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0 ? (float[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (float[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_complex_single(
            &x, mtxmatrix_coo, size, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxmatrix_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 3, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile.precision);
        TEST_ASSERT_EQ(3, mtxdistfile.datasize);
        const struct mtxfile_matrix_coordinate_complex_single * data =
            mtxdistfile.data.matrix_coordinate_complex_single;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(2, data[0].i);
            TEST_ASSERT_EQ(1.0f, data[0].a[0]);
            TEST_ASSERT_EQ(1.0f, data[0].a[1]);
            TEST_ASSERT_EQ(4, data[1].i);
            TEST_ASSERT_EQ(1.0f, data[1].a[0]);
            TEST_ASSERT_EQ(2.0f, data[1].a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(6, data[0].i);
            TEST_ASSERT_EQ(3.0f, data[0].a[0]);
            TEST_ASSERT_EQ(0.0f, data[0].a[1]);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0 ? (double[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (double[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_complex_double(
            &x, mtxmatrix_coo, size, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxmatrix_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 3, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(3, mtxdistfile.datasize);
        const struct mtxfile_matrix_coordinate_complex_double * data =
            mtxdistfile.data.matrix_coordinate_complex_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(2, data[0].i);
            TEST_ASSERT_EQ(1.0, data[0].a[0]);
            TEST_ASSERT_EQ(1.0, data[0].a[1]);
            TEST_ASSERT_EQ(4, data[1].i);
            TEST_ASSERT_EQ(1.0, data[1].a[0]);
            TEST_ASSERT_EQ(2.0, data[1].a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(6, data[0].i);
            TEST_ASSERT_EQ(3.0, data[0].a[0]);
            TEST_ASSERT_EQ(0.0, data[0].a[1]);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int32_t * xdata = rank == 0 ? (int32_t[2]) {1, 1} : (int32_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_integer_single(
            &x, mtxmatrix_coo, size, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxmatrix_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_matrix_coordinate_integer_single * data =
            mtxdistfile.data.matrix_coordinate_integer_single;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3, data[2].a);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxmatrix_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int64_t * xdata = rank == 0 ? (int64_t[2]) {1, 1} : (int64_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxmatrix_dist x;
        err = mtxmatrix_dist_init_integer_double(
            &x, mtxmatrix_coo, size, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxmatrix_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_matrix_coordinate_integer_double * data =
            mtxdistfile.data.matrix_coordinate_integer_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ( 6, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 8, data[1].i); TEST_ASSERT_EQ(2, data[1].a);
            TEST_ASSERT_EQ(10, data[2].i); TEST_ASSERT_EQ(3, data[2].a);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxmatrix_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}
#endif

/**
 * ‘test_mtxmatrix_dist_split()’ tests splitting matrices.
 */
int test_mtxmatrix_dist_split(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
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
        struct mtxmatrix_dist src;
        struct mtxmatrix_dist dst0, dst1;
        struct mtxmatrix_dist * dsts[] = {&dst0, &dst1};
        int num_parts = 2;
        int num_rows = 3;
        int num_columns = 3;
        int srcnnz = rank == 0 ? 2 : 3;
        int64_t * srcrowidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {1, 2, 2};
        int64_t * srccolidx = rank == 0 ? (int64_t[2]) {0, 2} : (int64_t[3]) {0, 0, 2};
        float * srcdata = rank == 0 ? (float[2]) {1.0f, 3.0f} : (float[3]) {5.0f, 7.0f, 9.0f};
        err = mtxmatrix_dist_init_entries_global_real_single(
            &src, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns,
            srcnnz, srcrowidx, srccolidx, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int * parts = rank == 0 ? (int[2]) {0, 0} : (int[3]) {0, 1, 1};
        err = mtxmatrix_dist_split(
            num_parts, dsts, &src, srcnnz, parts, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(3, dst0.num_rows);
        TEST_ASSERT_EQ(3, dst0.num_columns);
        TEST_ASSERT_EQ(3, dst0.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, dst0.rowmapsize);
            TEST_ASSERT_EQ(0, dst0.rowmap[0]);
            TEST_ASSERT_EQ(2, dst0.colmapsize);
            TEST_ASSERT_EQ(0, dst0.colmap[0]);
            TEST_ASSERT_EQ(2, dst0.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coo, dst0.Ap.type);
            const struct mtxmatrix_coo * Ap = &dst0.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(0, Ap->rowidx[1]); TEST_ASSERT_EQ(1, Ap->colidx[1]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(1.0f, a[0]);
            TEST_ASSERT_EQ(3.0f, a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, dst0.rowmapsize);
            TEST_ASSERT_EQ(1, dst0.rowmap[0]);
            TEST_ASSERT_EQ(1, dst0.colmapsize);
            TEST_ASSERT_EQ(0, dst0.colmap[0]);
            TEST_ASSERT_EQ(mtxmatrix_coo, dst0.Ap.type);
            const struct mtxmatrix_coo * Ap = &dst0.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(1, Ap->num_rows);
            TEST_ASSERT_EQ(1, Ap->num_columns);
            TEST_ASSERT_EQ(1, Ap->num_entries);
            TEST_ASSERT_EQ(1, Ap->num_nonzeros);
            TEST_ASSERT_EQ(1, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(5.0f, a[0]);
        }
        TEST_ASSERT_EQ(3, dst1.num_rows);
        TEST_ASSERT_EQ(3, dst1.num_columns);
        TEST_ASSERT_EQ(2, dst1.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(0, dst1.rowmapsize);
            TEST_ASSERT_EQ(0, dst1.colmapsize);
            TEST_ASSERT_EQ(mtxmatrix_coo, dst1.Ap.type);
            const struct mtxmatrix_coo * Ap = &dst1.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(0, Ap->num_rows);
            TEST_ASSERT_EQ(0, Ap->num_columns);
            TEST_ASSERT_EQ(0, Ap->num_entries);
            TEST_ASSERT_EQ(0, Ap->num_nonzeros);
            TEST_ASSERT_EQ(0, Ap->size);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, dst1.rowmapsize);
            TEST_ASSERT_EQ(2, dst1.rowmap[0]);
            TEST_ASSERT_EQ(2, dst1.colmapsize);
            TEST_ASSERT_EQ(0, dst1.colmap[0]);
            TEST_ASSERT_EQ(2, dst1.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coo, dst1.Ap.type);
            const struct mtxmatrix_coo * Ap = &dst1.Ap.storage.coo;
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
        mtxmatrix_dist_free(&dst1); mtxmatrix_dist_free(&dst0);
        mtxmatrix_dist_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxmatrix_dist_swap()’ tests swapping values of two matrices.
 */
int test_mtxmatrix_dist_swap(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * ycolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_real_single(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, ynnz, yrowidx, ycolidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, x.num_rows);
        TEST_ASSERT_EQ(3, x.num_columns);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.rowmapsize);
            TEST_ASSERT_EQ(0, x.rowmap[0]);
            TEST_ASSERT_EQ(1, x.rowmap[1]);
            TEST_ASSERT_EQ(1, x.colmapsize);
            TEST_ASSERT_EQ(0, x.colmap[0]);
            TEST_ASSERT_EQ(mtxmatrix_coo, x.Ap.type);
            const struct mtxmatrix_coo * Ap = &x.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(1, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(1, Ap->rowidx[1]); TEST_ASSERT_EQ(0, Ap->colidx[1]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(2.0f, a[0]);
            TEST_ASSERT_EQ(1.0f, a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, x.rowmapsize);
            TEST_ASSERT_EQ(1, x.rowmap[0]);
            TEST_ASSERT_EQ(2, x.rowmap[1]);
            TEST_ASSERT_EQ(2, x.colmapsize);
            TEST_ASSERT_EQ(0, x.colmap[0]);
            TEST_ASSERT_EQ(2, x.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coo, x.Ap.type);
            const struct mtxmatrix_coo * Ap = &x.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(1, Ap->colidx[0]);
            TEST_ASSERT_EQ(1, Ap->rowidx[1]); TEST_ASSERT_EQ(0, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(1, Ap->colidx[2]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(0.0f, a[0]);
            TEST_ASSERT_EQ(2.0f, a[1]);
            TEST_ASSERT_EQ(1.0f, a[2]);
        }
        TEST_ASSERT_EQ(3, y.num_rows);
        TEST_ASSERT_EQ(3, y.num_columns);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.rowmapsize);
            TEST_ASSERT_EQ(0, y.rowmap[0]);
            TEST_ASSERT_EQ(1, y.rowmap[1]);
            TEST_ASSERT_EQ(1, y.colmapsize);
            TEST_ASSERT_EQ(0, y.colmap[0]);
            TEST_ASSERT_EQ(mtxmatrix_coo, y.Ap.type);
            const struct mtxmatrix_coo * Ap = &y.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(1, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(1, Ap->rowidx[1]); TEST_ASSERT_EQ(0, Ap->colidx[1]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(1.0f, a[0]);
            TEST_ASSERT_EQ(1.0f, a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, y.rowmapsize);
            TEST_ASSERT_EQ(1, y.rowmap[0]);
            TEST_ASSERT_EQ(2, y.rowmap[1]);
            TEST_ASSERT_EQ(2, y.colmapsize);
            TEST_ASSERT_EQ(0, y.colmap[0]);
            TEST_ASSERT_EQ(2, y.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coo, y.Ap.type);
            const struct mtxmatrix_coo * Ap = &y.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(1, Ap->colidx[0]);
            TEST_ASSERT_EQ(1, Ap->rowidx[1]); TEST_ASSERT_EQ(0, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(1, Ap->colidx[2]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(1.0f, a[0]);
            TEST_ASSERT_EQ(2.0f, a[1]);
            TEST_ASSERT_EQ(3.0f, a[2]);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * ycolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_real_single(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, ynnz, yrowidx, ycolidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, y.num_rows);
        TEST_ASSERT_EQ(3, y.num_columns);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.rowmapsize);
            TEST_ASSERT_EQ(0, y.rowmap[0]);
            TEST_ASSERT_EQ(1, y.rowmap[1]);
            TEST_ASSERT_EQ(1, y.colmapsize);
            TEST_ASSERT_EQ(0, y.colmap[0]);
            TEST_ASSERT_EQ(mtxmatrix_coo, y.Ap.type);
            const struct mtxmatrix_coo * Ap = &y.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(1, Ap->num_columns);
            TEST_ASSERT_EQ(2, Ap->num_entries);
            TEST_ASSERT_EQ(2, Ap->num_nonzeros);
            TEST_ASSERT_EQ(2, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(0, Ap->colidx[0]);
            TEST_ASSERT_EQ(1, Ap->rowidx[1]); TEST_ASSERT_EQ(0, Ap->colidx[1]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(1.0f, a[0]);
            TEST_ASSERT_EQ(1.0f, a[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, y.rowmapsize);
            TEST_ASSERT_EQ(1, y.rowmap[0]);
            TEST_ASSERT_EQ(2, y.rowmap[1]);
            TEST_ASSERT_EQ(2, y.colmapsize);
            TEST_ASSERT_EQ(0, y.colmap[0]);
            TEST_ASSERT_EQ(2, y.colmap[1]);
            TEST_ASSERT_EQ(mtxmatrix_coo, y.Ap.type);
            const struct mtxmatrix_coo * Ap = &y.Ap.storage.coo;
            TEST_ASSERT_EQ(mtx_unsymmetric, Ap->symmetry);
            TEST_ASSERT_EQ(2, Ap->num_rows);
            TEST_ASSERT_EQ(2, Ap->num_columns);
            TEST_ASSERT_EQ(4, Ap->num_entries);
            TEST_ASSERT_EQ(3, Ap->num_nonzeros);
            TEST_ASSERT_EQ(3, Ap->size);
            TEST_ASSERT_EQ(0, Ap->rowidx[0]); TEST_ASSERT_EQ(1, Ap->colidx[0]);
            TEST_ASSERT_EQ(1, Ap->rowidx[1]); TEST_ASSERT_EQ(0, Ap->colidx[1]);
            TEST_ASSERT_EQ(1, Ap->rowidx[2]); TEST_ASSERT_EQ(1, Ap->colidx[2]);
            const float * a = Ap->a.data.real_single;
            TEST_ASSERT_EQ(1.0f, a[0]);
            TEST_ASSERT_EQ(2.0f, a[1]);
            TEST_ASSERT_EQ(3.0f, a[2]);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.Ap.storage.coo.a.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.Ap.storage.coo.a.data.real_single[2]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ( 8.0f, x.Ap.storage.coo.a.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.Ap.storage.coo.a.data.real_single[2]);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.Ap.storage.coo.a.data.real_double[1]);
            TEST_ASSERT_EQ(6.0f, x.Ap.storage.coo.a.data.real_double[2]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ( 8.0f, x.Ap.storage.coo.a.data.real_double[1]);
            TEST_ASSERT_EQ(12.0f, x.Ap.storage.coo.a.data.real_double[2]);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ(4.0f, x.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(0.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 8.0f, x.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        float as[2] = {2, 3};
        err = mtxmatrix_dist_cscal(as, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -4.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 20.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ(-16.0f, x.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 28.0f, x.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 24.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 36.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        double ad[2] = {2, 3};
        err = mtxmatrix_dist_zscal(ad, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -68.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(  28.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ(-116.0f, x.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ(   8.0f, x.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( -60.0f, x.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 144.0f, x.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ(2.0f, x.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ(4.0f, x.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(0.0f, x.Ap.storage.coo.a.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 4.0f, x.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 8.0f, x.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.Ap.storage.coo.a.data.complex_double[0][1]);
        }
        float as[2] = {2, 3};
        err = mtxmatrix_dist_cscal(as, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -4.0, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 20.0, x.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ(-16.0, x.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 28.0, x.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 24.0, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 36.0, x.Ap.storage.coo.a.data.complex_double[0][1]);
        }
        double ad[2] = {2, 3};
        err = mtxmatrix_dist_zscal(ad, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -68.0, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(  28.0, x.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ(-116.0, x.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ(   8.0, x.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( -60.0, x.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 144.0, x.Ap.storage.coo.a.data.complex_double[0][1]);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_real_single(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ( 3.0f, y.Ap.storage.coo.a.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 2.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ( 6.0f, y.Ap.storage.coo.a.data.real_single[1]);
            TEST_ASSERT_EQ( 7.0f, y.Ap.storage.coo.a.data.real_single[2]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ( 5.0f, y.Ap.storage.coo.a.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.Ap.storage.coo.a.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.Ap.storage.coo.a.data.real_single[2]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.Ap.storage.coo.a.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 9.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.Ap.storage.coo.a.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.Ap.storage.coo.a.data.real_single[2]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.Ap.storage.coo.a.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.Ap.storage.coo.a.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.Ap.storage.coo.a.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.Ap.storage.coo.a.data.real_single[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_real_double(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ( 3.0, y.Ap.storage.coo.a.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 2.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ( 6.0, y.Ap.storage.coo.a.data.real_double[1]);
            TEST_ASSERT_EQ( 7.0, y.Ap.storage.coo.a.data.real_double[2]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ( 5.0, y.Ap.storage.coo.a.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.Ap.storage.coo.a.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.Ap.storage.coo.a.data.real_double[2]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.Ap.storage.coo.a.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 9.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.Ap.storage.coo.a.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.Ap.storage.coo.a.data.real_double[2]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.Ap.storage.coo.a.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.Ap.storage.coo.a.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.Ap.storage.coo.a.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.Ap.storage.coo.a.data.real_double[2]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{2.0f,1.0f}, {0.0f,2.0f}})
            : ((float[1][2]) {{1.0f,0.0f}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ(2.0f, y.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ(6.0f, y.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(0.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 5.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 4.0f, y.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ(10.0f, y.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 9.0f, y.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ(22.0f, y.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
            TEST_ASSERT_EQ(19.0f, y.Ap.storage.coo.a.data.complex_single[1][0]);
            TEST_ASSERT_EQ(46.0f, y.Ap.storage.coo.a.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0f, y.Ap.storage.coo.a.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.Ap.storage.coo.a.data.complex_single[0][1]);
        }
        mtxmatrix_dist_free(&y);
        mtxmatrix_dist_free(&x);
    }
    {
        struct mtxmatrix_dist x;
        struct mtxmatrix_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0,1.0}, {1.0,2.0}})
            : ((double[1][2]) {{3.0,0.0}});
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{2.0,1.0}, {0.0,2.0}})
            : ((double[1][2]) {{1.0,0.0}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ(2.0, y.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ(6.0, y.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(0.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 5.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 4.0, y.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ(10.0, y.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 9.0, y.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ(22.0, y.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
        }
        err = mtxmatrix_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
            TEST_ASSERT_EQ(19.0, y.Ap.storage.coo.a.data.complex_double[1][0]);
            TEST_ASSERT_EQ(46.0, y.Ap.storage.coo.a.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0, y.Ap.storage.coo.a.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.Ap.storage.coo.a.data.complex_double[0][1]);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        float * ydata = rank == 0 ? (float[2]) {3.0f, 2.0f} : (float[3]) {1.0f, 0.0f, 1.0f};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_real_single(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        double * ydata = rank == 0 ? (double[2]) {3.0, 2.0} : (double[3]) {1.0, 0.0, 1.0};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_real_double(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{3.0f,2.0f}, {1.0f,0.0f}})
            : ((float[1][2]) {{1.0f,0.0f}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * rowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * colidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0,1.0}, {1.0,2.0}})
            : ((double[1][2]) {{3.0,0.0}});
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{3.0,2.0}, {1.0,0.0}})
            : ((double[1][2]) {{1.0,0.0}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &y, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, nnz, rowidx, colidx, ydata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        float * xdata = rank == 0 ? (float[2]) {-1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[3]) {1, 2, 2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[3]) {2, 0, 2};
        double * xdata = rank == 0 ? (double[2]) {-1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxmatrix_dist_init_entries_global_real_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{-1.0f,-1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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
        int num_rows = 3;
        int num_columns = 3;
        int64_t * xrowidx = rank == 0 ? (int64_t[2]) {0, 1} : (int64_t[1]) {2};
        int64_t * xcolidx = rank == 0 ? (int64_t[2]) {0, 0} : (int64_t[1]) {2};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{-1.0f,-1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &x, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, xnnz, xrowidx, xcolidx, xdata, comm, &disterr);
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

/**
 * ‘test_mtxmatrix_dist_gemv()’ tests computing matrix-vector
 * products.
 */
int test_mtxmatrix_dist_gemv(void)
{
    int err;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    MPI_Comm comm = MPI_COMM_WORLD;
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

    /*
     * a) For unsymmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 0⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 14⎤  ⎡ 3⎤  ⎡ 17⎤
     * 2*⎢ 4 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 56⎥ +⎢ 0⎥ =⎢ 56⎥,
     *   ⎣ 7 8 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 92⎦  ⎣ 6⎦  ⎣ 98⎦
     *
     * and the transposed product
     *
     *   ⎡ 1 4 7⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 36⎤  ⎡ 3⎤  ⎡ 39⎤
     * 2*⎢ 2 5 8⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 48⎥ +⎢ 0⎥ =⎢ 48⎥.
     *   ⎣ 0 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 42⎦  ⎣ 6⎦  ⎣ 48⎦
     *
     * b) For symmetric real or integer matrices, calculate
     *
     *   ⎡ 1 2 0⎤  ⎡ 3⎤     ⎡ 1⎤  ⎡ 14⎤  ⎡ 3⎤  ⎡ 17⎤
     * 2*⎢ 2 5 6⎥ *⎢ 2⎥ + 3*⎢ 0⎥ =⎢ 44⎥ +⎢ 0⎥ =⎢ 44⎥,
     *   ⎣ 0 6 9⎦  ⎣ 1⎦     ⎣ 2⎦  ⎣ 42⎦  ⎣ 6⎦  ⎣ 48⎦
     *
     * c) For unsymmetric complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-8+34i⎤  ⎡ 3   ⎤  ⎡-5+34i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 0+90i⎦  ⎣ 6+6i⎦  ⎣ 6+96i⎦
     *
     * and the transposed product
     *
     *   ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-12+46i⎤  ⎡ 3   ⎤  ⎡-9+46i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ -8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * and the conjugate transposed product
     *
     *   ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 44-2i⎤  ⎡ 3   ⎤  ⎡ 47-2i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72-6i⎦  ⎣ 6+6i⎦  ⎣ 78   ⎦
     *
     * and the product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34-8i⎤  ⎡ 3+1i⎤  ⎡-31-7i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥,
     *    ⎣ 5+6i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-90   ⎦  ⎣ 4+8i⎦  ⎣-86+8i⎦
     *
     * and the transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+2i 5+6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-46-12i⎤  ⎡ 3+1i⎤  ⎡-43-11i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥,
     *    ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74- 8i⎦  ⎣ 4+8i⎦  ⎣-70    ⎦
     *
     * and the conjugate transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1-2i 5-6i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 2+44i⎤  ⎡ 3+1i⎤  ⎡  5+45i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢       ⎥.
     *    ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣ 6+72i⎦  ⎣ 4+8i⎦  ⎣ 10+80i⎦
     *
     * d) For symmetric complex matrices, calculate
     *
     *   ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-8+34i⎤  ⎡ 3   ⎤  ⎡-5+34i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣-8+74i⎦  ⎣ 6+6i⎦  ⎣-2+80i⎦
     *
     * and the conjugate transposed product
     *
     *   ⎡ 1-2i 3-4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡ 32-6i⎤  ⎡ 3   ⎤  ⎡ 35-6i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 72-6i⎦  ⎣ 6+6i⎦  ⎣ 78+0i⎦
     *
     * and the product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+2i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡-34-8i⎤  ⎡ 3+1i⎤  ⎡-31-7i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *    ⎣ 3+4i 7+8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-74-8i⎦  ⎣ 4+8i⎦  ⎣-70+0i⎦
     *
     * and the conjugate transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1-2i 3-4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ 6+32i⎤  ⎡ 3+1i⎤  ⎡ 9+33i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *    ⎣ 3-4i 7-8i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣ 6+72i⎦  ⎣ 4+8i⎦  ⎣10+80i⎦
     *
     * e) for Hermitian complex matrices, calculate
     *
     *   ⎡ 1+0i 3+4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡-4+22i⎤  ⎡ 3   ⎤  ⎡-1+22i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢      ⎥ +⎢     ⎥ =⎢      ⎥.
     *   ⎣ 3-4i 7+0i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣40+10i⎦  ⎣ 6+6i⎦  ⎣46+16i⎦
     *
     * and the transposed product
     *
     *   ⎡ 1+0i 3-4i⎤  ⎡ 3+1i⎤     ⎡ 1+0i⎤  ⎡  28+6i⎤  ⎡ 3   ⎤  ⎡  31+6i⎤
     * 2*⎢          ⎥ *⎢     ⎥ + 3*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥.
     *   ⎣ 3+4i 7+0i⎦  ⎣ 1+2i⎦     ⎣ 2+2i⎦  ⎣ 24+58i⎦  ⎣ 6+6i⎦  ⎣ 30+64i⎦
     *
     * and the product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+0i 3+4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ -22-4i⎤  ⎡ 3+1i⎤  ⎡-19-3i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢      ⎥.
     *    ⎣ 3-4i 7+0i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-10+40i⎦  ⎣ 4+8i⎦  ⎣-6+48i⎦
     *
     * and the transposed product with complex coefficients (cgemv/zgemv)
     *
     *    ⎡ 1+0i 3-4i⎤  ⎡ 3+1i⎤          ⎡ 1+0i⎤  ⎡ -6+28i⎤  ⎡ 3+1i⎤  ⎡ -3+29i⎤
     * 2i*⎢          ⎥ *⎢     ⎥ + (3+1i)*⎢     ⎥ =⎢       ⎥ +⎢     ⎥ =⎢       ⎥.
     *    ⎣ 3+4i 7+0i⎦  ⎣ 1+2i⎦          ⎣ 2+2i⎦  ⎣-58+24i⎦  ⎣ 4+8i⎦  ⎣-54+32i⎦
     *
     */

    /*
     * Real, single precision, unsymmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        /* int Arowidx[] = {0, 0, 1, 1, 1, 2, 2, 2}; */
        /* int Acolidx[] = {0, 1, 0, 1, 2, 0, 1, 2}; */
        /* float * Adata = {1.0f, 2.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}; */
        /* float xdata[] = {3.0f, 2.0f, 1.0f}; */
        /* float ydata[] = {1.0f, 0.0f, 2.0f}; */

        int64_t * Arowidx = rank==0 ? (int64_t[5]){0,0,1,1,1} : (int64_t[3]){2,2,2};
        int64_t * Acolidx = rank==0 ? (int64_t[5]){0,1,0,1,2} : (int64_t[3]){0,1,2};
        float * Adata = rank==0 ? (float[5]){1,2,4,5,6} : (float[3]) {7,8,9};
        int Annz = rank == 0 ? 5 : 3;
        int64_t * xidx = rank==0 ? (int64_t[2]){0, 1} : (int64_t[1]){2};
        float * xdata = rank==0 ? (float[2]){3,2} : (float[1]){1};
        int xnnz = rank == 0 ? 2 : 1;
        int64_t * yidx = rank==0 ? (int64_t[2]){0, 1} : (int64_t[1]){2};
        float * ydata = rank==0 ? (float[2]){1,0} : (float[1]){2};
        int ynnz = rank == 0 ? 2 : 1;
        err = mtxmatrix_dist_init_entries_global_real_single(
            &A, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 98.0f);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(
                MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
                ? mtxdisterror_description(&disterr) : mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 48.0f);
            }
            mtxvector_dist_free(&y);
        }
        /* { */
        /*     err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_base, y.xp.type); */
        /*     const struct mtxvector_base * y_ = &y.xp.storage.base; */
        /*     TEST_ASSERT_EQ(mtx_field_real, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     if (rank == 0) { */
        /*         TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[1], 56.0f); */
        /*     } else if (rank == 1) { */
        /*         TEST_ASSERT_EQ(1, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[0], 98.0f); */
        /*     } */
        /*     mtxvector_dist_free(&y); */
        /* } */
        /* { */
        /*     err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr); */
        /*     TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err)); */
        /*     TEST_ASSERT_EQ(mtxvector_base, y.xp.type); */
        /*     const struct mtxvector_base * y_ = &y.xp.storage.base; */
        /*     TEST_ASSERT_EQ(mtx_field_real, y_->field); */
        /*     TEST_ASSERT_EQ(mtx_single, y_->precision); */
        /*     if (rank == 0) { */
        /*         TEST_ASSERT_EQ(2, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[0], 39.0f); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[1], 48.0f); */
        /*     } else if (rank == 1) { */
        /*         TEST_ASSERT_EQ(1, y_->size); */
        /*     TEST_ASSERT_EQ(y_->data.real_single[0], 48.0f); */
        /*     } */
        /*     mtxvector_dist_free(&y); */
        /* } */
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }
#if 0
    /*
     * Real, single precision, symmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        float Adata[] = {1.0f, 2.0f, 5.0f, 6.0f, 9.0f};
        float xdata[] = {3.0f, 2.0f, 1.0f};
        float ydata[] = {1.0f, 0.0f, 2.0f};
        err = mtxmatrix_dist_init_entries_global_real_single(
            &A, mtxmatrix_coo, mtx_symmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 48.0f);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 48.0f);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 48.0f);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 17.0f);
                TEST_ASSERT_EQ(y_->data.real_single[1], 44.0f);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_single[0], 48.0f);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Real, double precision, unsymmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 8;
        int64_t * Arowidx = rank==0 ? (int64_t[5]){0,0,1,1,1} : (int64_t[3]){2,2,2};
        int64_t * Acolidx = rank==0 ? (int64_t[5]){0,1,0,1,2} : (int64_t[3]){0,1,2};
        double Adata[] = {1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_dist_init_entries_global_real_double(
            &A, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 98.0);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 48.0);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 56.0);
                TEST_ASSERT_EQ(y_->data.real_double[0], 98.0);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 39.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 48.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 48.0);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Real, double precision, symmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 5;
        int Arowidx[] = {0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 1, 2, 2};
        double Adata[] = {1.0, 2.0, 5.0, 6.0, 9.0};
        double xdata[] = {3.0, 2.0, 1.0};
        double ydata[] = {1.0, 0.0, 2.0};
        err = mtxmatrix_dist_init_entries_global_real_double(
            &A, mtxmatrix_coo, mtx_symmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 48.0);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 48.0);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 48.0);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_real_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_real, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 17.0);
                TEST_ASSERT_EQ(y_->data.real_double[1], 44.0);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.real_double[0], 48.0);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Complex, single precision, unsymmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 4;
        int Arowidx[] = {0, 0, 1, 1};
        int Acolidx[] = {0, 1, 0, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {5.0f,6.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &A, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 96.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 47.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 96.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 47.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_dist_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -86.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   8.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -43.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 45.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -86.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   8.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -43.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -11.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 45.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Complex, single precision, symmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        float Adata[][2] = {{1.0f,2.0f}, {3.0f,4.0f}, {7.0f,8.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &A, mtxmatrix_coo, mtx_symmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 35.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -5.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 34.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -2.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 35.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 78.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  0.0f);
            mtxvector_dist_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 33.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -7.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -70.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],   0.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  9.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 33.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 10.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 80.0f);
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Complex, single precision, hermitian matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        float Adata[][2] = {{1.0f,0.0f}, {3.0f,4.0f}, {7.0f,0.0f}};
        float xdata[][2] = {{3.0f,1.0f}, {1.0f,2.0f}};
        float ydata[][2] = {{1.0f,0.0f}, {2.0f,2.0f}};
        err = mtxmatrix_dist_init_entries_global_complex_single(
            &A, mtxmatrix_coo, mtx_hermitian, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 30.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 64.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], 31.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 30.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 64.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -1.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], 22.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], 46.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 16.0f);
            mtxvector_dist_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  48.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  29.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -54.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  32.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],-19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1], -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1], 48.0f);
            mtxvector_dist_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  48.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  29.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0], -54.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  32.0f);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_single[0][0], -19.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[0][1],  -3.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][0],  -6.0f);
            TEST_ASSERT_EQ(y_->data.complex_single[1][1],  48.0f);
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Complex, double precision, unsymmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 4;
        int Arowidx[] = {0, 0, 1, 1};
        int Acolidx[] = {0, 1, 0, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {5.0,6.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &A, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 96.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 47.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 96.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 47.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_dist_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -86.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   8.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -43.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -11.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 45.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -86.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   8.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -43.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -11.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 45.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Complex, double precision, symmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        double Adata[][2] = {{1.0,2.0}, {3.0,4.0}, {7.0,8.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &A, mtxmatrix_coo, mtx_symmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 35.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -5.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 34.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -2.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 35.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 78.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  0.0);
            mtxvector_dist_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 33.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -7.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -70.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],   0.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  9.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 33.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 10.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 80.0);
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Complex, double precision, hermitian matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 2;
        int num_columns = 2;
        int num_nonzeros = 3;
        int Arowidx[] = {0, 0, 1};
        int Acolidx[] = {0, 1, 1};
        double Adata[][2] = {{1.0,0.0}, {3.0,4.0}, {7.0,0.0}};
        double xdata[][2] = {{3.0,1.0}, {1.0,2.0}};
        double ydata[][2] = {{1.0,0.0}, {2.0,2.0}};
        err = mtxmatrix_dist_init_entries_global_complex_double(
            &A, mtxmatrix_coo, mtx_hermitian, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 30.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 64.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_conjtrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], 31.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 30.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 64.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_conjtrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -1.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], 22.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], 46.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 16.0);
            mtxvector_dist_free(&y);
        }
        float calpha[2] = {0.0f, 2.0f};
        float cbeta[2]  = {3.0f, 1.0f};
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_notrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  48.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_trans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  29.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -54.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  32.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_cgemv(mtx_conjtrans, calpha, &A, &x, cbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],-19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1], -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1], 48.0);
            mtxvector_dist_free(&y);
        }
        double zalpha[2] = {0.0, 2.0};
        double zbeta[2]  = {3.0, 1.0};
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_notrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  48.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_trans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  29.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0], -54.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  32.0);
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_complex_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_zgemv(mtx_conjtrans, zalpha, &A, &x, zbeta, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_complex, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            TEST_ASSERT_EQ(2, y_->size);
            TEST_ASSERT_EQ(y_->data.complex_double[0][0], -19.0);
            TEST_ASSERT_EQ(y_->data.complex_double[0][1],  -3.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][0],  -6.0);
            TEST_ASSERT_EQ(y_->data.complex_double[1][1],  48.0);
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Integer, single precision, unsymmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 9;
        int Arowidx[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        int32_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_dist_init_entries_global_integer_single(
            &A, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_integer_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 98);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 98);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Integer, single precision, symmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 6;
        int Arowidx[] = {0, 0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 2, 1, 2, 2};
        int32_t Adata[] = {1, 2, 3, 5, 6, 9};
        int32_t xdata[] = {3, 2, 1};
        int32_t ydata[] = {1, 0, 2};
        err = mtxmatrix_dist_init_entries_global_integer_single(
            &A, mtxmatrix_coo, mtx_symmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_integer_single(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_single(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_single, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_single[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_single[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Integer, double precision, unsymmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 9;
        int Arowidx[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        int Acolidx[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        int64_t Adata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_dist_init_entries_global_integer_double(
            &A, mtxmatrix_coo, mtx_unsymmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_integer_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 98);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 56);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 98);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 39);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 48);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }

    /*
     * Integer, double precision, symmetric matrices
     */

    {
        struct mtxmatrix_dist A;
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 6;
        int Arowidx[] = {0, 0, 0, 1, 1, 2};
        int Acolidx[] = {0, 1, 2, 1, 2, 2};
        int64_t Adata[] = {1, 2, 3, 5, 6, 9};
        int64_t xdata[] = {3, 2, 1};
        int64_t ydata[] = {1, 0, 2};
        err = mtxmatrix_dist_init_entries_global_integer_double(
            &A, mtxmatrix_coo, mtx_symmetric, num_rows, num_columns, Annz, Arowidx, Acolidx, Adata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_integer_double(&x, mtxvector_base, num_columns, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_notrans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_sgemv(mtx_trans, 2.0f, &A, &x, 3.0f, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_notrans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        {
            err = mtxvector_dist_init_integer_double(&y, mtxvector_base, num_rows, ynnz, yidx, ydata, comm, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            err = mtxmatrix_dist_dgemv(mtx_trans, 2.0, &A, &x, 3.0, &y, NULL, &disterr);
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
            TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
            const struct mtxvector_base * y_ = &y.xp.storage.base;
            TEST_ASSERT_EQ(mtx_field_integer, y_->field);
            TEST_ASSERT_EQ(mtx_double, y_->precision);
            if (rank == 0) {
                TEST_ASSERT_EQ(2, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 23);
                TEST_ASSERT_EQ(y_->data.integer_double[1], 44);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, y_->size);
                TEST_ASSERT_EQ(y_->data.integer_double[0], 66);
            }
            mtxvector_dist_free(&y);
        }
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
    }
#endif
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;

    /* 1. initialise MPI. */
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
    TEST_RUN(test_mtxmatrix_dist_to_mtxfile);
    TEST_RUN(test_mtxmatrix_dist_from_mtxdistfile);
    /* TEST_RUN(test_mtxmatrix_dist_to_mtxdistfile); */
    TEST_RUN(test_mtxmatrix_dist_split);
    TEST_RUN(test_mtxmatrix_dist_swap);
    TEST_RUN(test_mtxmatrix_dist_copy);
    TEST_RUN(test_mtxmatrix_dist_scal);
    TEST_RUN(test_mtxmatrix_dist_axpy);
    TEST_RUN(test_mtxmatrix_dist_dot);
    TEST_RUN(test_mtxmatrix_dist_nrm2);
    TEST_RUN(test_mtxmatrix_dist_asum);
    TEST_RUN(test_mtxmatrix_dist_gemv);
    TEST_SUITE_END();

    /* 3. clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
