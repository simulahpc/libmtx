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
 * Unit tests for distributed vectors.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/dist.h>
#include <libmtx/vector/vector.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char * program_invocation_short_name = "test_mtxvector_dist";

/**
 * ‘test_mtxvector_dist_from_mtxfile()’ tests converting Matrix
 *  Market files to vectors.
 */
int test_mtxvector_dist_from_mtxfile(void)
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
        int size = 9;
        struct mtxfile_vector_coordinate_real_single mtxdata[] = {
            {1, 1.0f}, {2, 2.0f}, {4, 4.0f}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_real_single(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(9, xdist.size);
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(0, xdist.xp.storage.base.idx[0]);
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.idx[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(3, xdist.xp.storage.base.idx[0]);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * xbase = &xdist.xp.storage.base;
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
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 9;
        struct mtxfile_vector_coordinate_real_double mtxdata[] = {
            {1, 1.0}, {2, 2.0}, {4, 4.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_real_double(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[1], 1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 3);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * x = &xdist.xp.storage.base;
        TEST_ASSERT_EQ(mtx_field_real, x->field);
        TEST_ASSERT_EQ(mtx_double, x->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x->size);
            TEST_ASSERT_EQ(x->data.real_double[0], 1.0);
            TEST_ASSERT_EQ(x->data.real_double[1], 2.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, x->size);
            TEST_ASSERT_EQ(x->data.real_double[0], 4.0);
        }
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 9;
        struct mtxfile_vector_coordinate_complex_single mtxdata[] = {
            {1,{1.0f,-1.0f}}, {2,{2.0f,-2.0f}}, {4,{4.0f,-4.0f}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_complex_single(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[1], 1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 3);
            TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        }
        const struct mtxvector_base * x = &xdist.xp.storage.base;
        TEST_ASSERT_EQ(mtx_field_complex, x->field);
        TEST_ASSERT_EQ(mtx_single, x->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x->size);
            TEST_ASSERT_EQ(x->data.complex_single[0][0], 1.0f);
            TEST_ASSERT_EQ(x->data.complex_single[0][1], -1.0f);
            TEST_ASSERT_EQ(x->data.complex_single[1][0], 2.0f);
            TEST_ASSERT_EQ(x->data.complex_single[1][1], -2.0f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, x->size);
            TEST_ASSERT_EQ(x->data.complex_single[0][0], 4.0f);
            TEST_ASSERT_EQ(x->data.complex_single[0][1], -4.0f);
        }
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 9;
        struct mtxfile_vector_coordinate_complex_double mtxdata[] = {
            {1,{1.0,-1.0}}, {2,{2.0,-2.0}}, {4,{4.0,-4.0}}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_complex_double(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[1], 1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 3);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * x = &xdist.xp.storage.base;
        TEST_ASSERT_EQ(mtx_field_complex, x->field);
        TEST_ASSERT_EQ(mtx_double, x->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x->size);
            TEST_ASSERT_EQ(x->data.complex_double[0][0], 1.0);
            TEST_ASSERT_EQ(x->data.complex_double[0][1], -1.0);
            TEST_ASSERT_EQ(x->data.complex_double[1][0], 2.0);
            TEST_ASSERT_EQ(x->data.complex_double[1][1], -2.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, x->size);
            TEST_ASSERT_EQ(x->data.complex_double[0][0], 4.0);
            TEST_ASSERT_EQ(x->data.complex_double[0][1], -4.0);
        }
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 9;
        struct mtxfile_vector_coordinate_integer_single mtxdata[] = {
            {1, 1}, {2, 2}, {4, 4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[1], 1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 3);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * x = &xdist.xp.storage.base;
        TEST_ASSERT_EQ(mtx_field_integer, x->field);
        TEST_ASSERT_EQ(mtx_single, x->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x->size);
            TEST_ASSERT_EQ(x->data.integer_single[0], 1);
            TEST_ASSERT_EQ(x->data.integer_single[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, x->size);
            TEST_ASSERT_EQ(x->data.integer_single[0], 4);
        }
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 9;
        struct mtxfile_vector_coordinate_integer_double mtxdata[] = {
            {1,1}, {2,2}, {4,4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_integer_double(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[1], 1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 3);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * x = &xdist.xp.storage.base;
        TEST_ASSERT_EQ(mtx_field_integer, x->field);
        TEST_ASSERT_EQ(mtx_double, x->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x->size);
            TEST_ASSERT_EQ(x->data.integer_double[0], 1);
            TEST_ASSERT_EQ(x->data.integer_double[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, x->size);
            TEST_ASSERT_EQ(x->data.integer_double[0], 4);
        }
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    {
        int size = 9;
        struct mtxfile_vector_coordinate_pattern mtxdata[] = {{1}, {2}, {4}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile mtxfile;
        err = mtxfile_init_vector_coordinate_pattern(
            &mtxfile, size, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxfile(
            &xdist, &mtxfile, mtxvector_base, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[1], 1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(xdist.xp.storage.base.idx[0], 3);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * x = &xdist.xp.storage.base;
        TEST_ASSERT_EQ(mtx_field_pattern, x->field);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x->size);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, x->size);
        }
        mtxvector_dist_free(&xdist);
        mtxfile_free(&mtxfile);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_to_mtxfile()’ tests converting vectors to
 * Matrix Market files.
 */
int test_mtxvector_dist_to_mtxfile(void)
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
        struct mtxvector_dist x;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ(5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
            const struct mtxfile_vector_coordinate_real_single * data =
                mtxfile.data.vector_coordinate_real_single;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0f, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0f, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1.0f, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2.0f, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3.0f, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0f, 1.0f} : (double[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ(5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
            const struct mtxfile_vector_coordinate_real_double * data =
                mtxfile.data.vector_coordinate_real_double;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1.0, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1.0, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2.0, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3.0, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0 ? (float[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (float[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 3, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
            const struct mtxfile_vector_coordinate_complex_single * data =
                mtxfile.data.vector_coordinate_complex_single;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0 ? (double[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (double[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 3, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
            const struct mtxfile_vector_coordinate_complex_double * data =
                mtxfile.data.vector_coordinate_complex_double;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int32_t * xdata = rank == 0 ? (int32_t[2]) {1, 1} : (int32_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_integer_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
            const struct mtxfile_vector_coordinate_integer_single * data =
                mtxfile.data.vector_coordinate_integer_single;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int64_t * xdata = rank == 0 ? (int64_t[2]) {1, 1} : (int64_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_integer_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxfile mtxfile;
        err = mtxvector_dist_to_mtxfile(&mtxfile, &x, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
            TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
            TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
            TEST_ASSERT_EQ(size, mtxfile.size.num_rows);
            TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
            TEST_ASSERT_EQ( 5, mtxfile.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, mtxfile.precision);
            const struct mtxfile_vector_coordinate_integer_double * data =
                mtxfile.data.vector_coordinate_integer_double;
            TEST_ASSERT_EQ( 2, data[0].i); TEST_ASSERT_EQ(1, data[0].a);
            TEST_ASSERT_EQ( 4, data[1].i); TEST_ASSERT_EQ(1, data[1].a);
            TEST_ASSERT_EQ( 6, data[2].i); TEST_ASSERT_EQ(1, data[2].a);
            TEST_ASSERT_EQ( 8, data[3].i); TEST_ASSERT_EQ(2, data[3].a);
            TEST_ASSERT_EQ(10, data[4].i); TEST_ASSERT_EQ(3, data[4].a);
            mtxfile_free(&mtxfile);
        }
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_from_mtxdistfile()’ tests converting
 *  distributed Matrix Market files to vectors.
 */
int test_mtxvector_dist_from_mtxdistfile(void)
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
        int num_rows = 9;
        const struct mtxfile_vector_coordinate_real_single * srcdata = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_single[2])
                {{1,1.0f}, {2,2.0f}})
            : ((const struct mtxfile_vector_coordinate_real_single[1])
                {{4,4.0f}});
        int64_t num_nonzeros = 3;
        int64_t localdatasize = rank == 0 ? 2 : 1;

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_coordinate_real_single(
            &src, num_rows, num_nonzeros, localdatasize, NULL,
            srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxvector_dist xdist;
        err = mtxvector_dist_from_mtxdistfile(
            &xdist, &src, mtxvector_base, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(9, xdist.size);
        TEST_ASSERT_EQ(9, xdist.xp.storage.base.size);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(0, xdist.xp.storage.base.idx[0]);
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.idx[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, xdist.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(3, xdist.xp.storage.base.idx[0]);
        }
        TEST_ASSERT_EQ(mtxvector_base, xdist.xp.type);
        const struct mtxvector_base * xbase = &xdist.xp.storage.base;
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
        mtxvector_dist_free(&xdist);
        mtxdistfile_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_to_mtxdistfile()’ tests converting vectors to
 * distributed Matrix Market files.
 */
int test_mtxvector_dist_to_mtxdistfile(void)
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
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxvector_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_vector_coordinate_real_single * data =
            mtxdistfile.data.vector_coordinate_real_single;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0f, 1.0f} : (double[3]) {1.0f, 2.0f, 3.0f};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxvector_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_vector_coordinate_real_double * data =
            mtxdistfile.data.vector_coordinate_real_double;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0 ? (float[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (float[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxvector_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 3, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile.precision);
        TEST_ASSERT_EQ(3, mtxdistfile.datasize);
        const struct mtxfile_vector_coordinate_complex_single * data =
            mtxdistfile.data.vector_coordinate_complex_single;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0 ? (double[2][2]) {{1.0f, 1.0f}, {1.0f, 2.0f}} : (double[1][2]) {{3.0f, 0.0f}};
        int nnz = rank == 0 ? 2 : 1;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxvector_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 3, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(3, mtxdistfile.datasize);
        const struct mtxfile_vector_coordinate_complex_double * data =
            mtxdistfile.data.vector_coordinate_complex_double;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int32_t * xdata = rank == 0 ? (int32_t[2]) {1, 1} : (int32_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_integer_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxvector_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_single, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtxdistfile.data.vector_coordinate_integer_single;
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
        mtxvector_dist_free(&x);
    }
    {
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        int64_t * xdata = rank == 0 ? (int64_t[2]) {1, 1} : (int64_t[3]) {1, 2, 3};
        int nnz = rank == 0 ? 2 : 3;
        struct mtxvector_dist x;
        err = mtxvector_dist_init_integer_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        struct mtxdistfile mtxdistfile;
        err = mtxvector_dist_to_mtxdistfile(
            &mtxdistfile, &x, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(size, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ( 5, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.datasize);
        const struct mtxfile_vector_coordinate_integer_double * data =
            mtxdistfile.data.vector_coordinate_integer_double;
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
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_split()’ tests splitting vectors.
 */
int test_mtxvector_dist_split(void)
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
        struct mtxvector_dist src;
        struct mtxvector_dist dst0, dst1;
        struct mtxvector_dist * dsts[] = {&dst0, &dst1};
        int num_parts = 2;
        int size = 12;
        int srcnnz = rank == 0 ? 2 : 3;
        int64_t * srcidx = rank == 0 ? (int64_t[2]) {1, 3} : (int64_t[3]) {5, 7, 9};
        float * srcdata = rank == 0 ? (float[2]) {1.0f, 3.0f} : (float[3]) {5.0f, 7.0f, 9.0f};
        err = mtxvector_dist_init_real_single(
            &src, mtxvector_base, size, srcnnz, srcidx, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int * parts = rank == 0 ? (int[2]) {0, 0} : (int[3]) {0, 1, 1};
        err = mtxvector_dist_split(
            num_parts, dsts, &src, srcnnz, parts, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, dst0.size);
        TEST_ASSERT_EQ(3, dst0.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(12, dst0.xp.storage.base.size);
            TEST_ASSERT_EQ(2, dst0.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(1, dst0.xp.storage.base.idx[0]);
            TEST_ASSERT_EQ(3, dst0.xp.storage.base.idx[1]);
            TEST_ASSERT_EQ(mtxvector_base, dst0.xp.type);
            TEST_ASSERT_EQ(mtx_field_real, dst0.xp.storage.base.field);
            TEST_ASSERT_EQ(mtx_single, dst0.xp.storage.base.precision);
            TEST_ASSERT_EQ(2, dst0.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0f, dst0.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(3.0f, dst0.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12, dst0.xp.storage.base.size);
            TEST_ASSERT_EQ(1, dst0.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(5, dst0.xp.storage.base.idx[0]);
            TEST_ASSERT_EQ(mtxvector_base, dst0.xp.type);
            TEST_ASSERT_EQ(mtx_field_real, dst0.xp.storage.base.field);
            TEST_ASSERT_EQ(mtx_single, dst0.xp.storage.base.precision);
            TEST_ASSERT_EQ(1, dst0.xp.storage.base.size);
            TEST_ASSERT_EQ(5.0f, dst0.xp.storage.base.data.real_single[0]);
        }
        TEST_ASSERT_EQ(12, dst1.size);
        TEST_ASSERT_EQ(2, dst1.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(12, dst1.xp.storage.base.size);
            TEST_ASSERT_EQ(0, dst1.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(mtxvector_base, dst1.xp.type);
            TEST_ASSERT_EQ(mtx_field_real, dst1.xp.storage.base.field);
            TEST_ASSERT_EQ(mtx_single, dst1.xp.storage.base.precision);
            TEST_ASSERT_EQ(0, dst1.xp.storage.base.size);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12, dst1.xp.storage.base.size);
            TEST_ASSERT_EQ(2, dst1.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(7, dst1.xp.storage.base.idx[0]);
            TEST_ASSERT_EQ(9, dst1.xp.storage.base.idx[1]);
            TEST_ASSERT_EQ(mtxvector_base, dst1.xp.type);
            TEST_ASSERT_EQ(mtx_field_real, dst1.xp.storage.base.field);
            TEST_ASSERT_EQ(mtx_single, dst1.xp.storage.base.precision);
            TEST_ASSERT_EQ(2, dst1.xp.storage.base.size);
            TEST_ASSERT_EQ(7.0f, dst1.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(9.0f, dst1.xp.storage.base.data.real_single[1]);
        }
        mtxvector_dist_free(&dst1); mtxvector_dist_free(&dst0);
        mtxvector_dist_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_swap()’ tests swapping values of two
 * vectors.
 */
int test_mtxvector_dist_swap(void)
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
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.storage.base.size);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, x.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, x.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, x.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.size);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(1.0f, x.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.size);
            TEST_ASSERT_EQ(0.0f, x.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(1.0f, x.xp.storage.base.data.real_single[2]);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(1.0f, y.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, y.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(3.0f, y.xp.storage.base.data.real_single[2]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.storage.base.size);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, x.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, x.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, x.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.size);
            TEST_ASSERT_EQ(2.0, x.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(1.0, x.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.size);
            TEST_ASSERT_EQ(0.0, x.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, x.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(1.0, x.xp.storage.base.data.real_double[2]);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(1.0, y.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, y.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(3.0, y.xp.storage.base.data.real_double[2]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
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
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.storage.base.size);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, x.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, x.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, x.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.size);
            TEST_ASSERT_EQ( 2.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.size);
            TEST_ASSERT_EQ( 0.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 2.0f, x.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.storage.base.data.complex_single[2][1]);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.storage.base.data.complex_single[2][1]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
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
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_swap(&x, &y, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, x.size);
        TEST_ASSERT_EQ(12, x.xp.storage.base.size);
        TEST_ASSERT_EQ(5, x.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 1);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[0], 4);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(x.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, x.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, x.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, x.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, x.xp.storage.base.size);
            TEST_ASSERT_EQ( 2.0f, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, x.xp.storage.base.size);
            TEST_ASSERT_EQ( 0.0f, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 2.0f, x.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-2.0f, x.xp.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 1.0f, x.xp.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-1.0f, x.xp.storage.base.data.complex_double[2][1]);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.storage.base.data.complex_double[2][1]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_copy()’ tests copying values from one vector
 * to another.
 */
int test_mtxvector_dist_copy(void)
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
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(1.0f, y.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, y.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(3.0f, y.xp.storage.base.data.real_single[2]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * yidx = rank == 0 ? (int64_t[2]) {1, 2} : (int64_t[3]) {4, 6, 9};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int ynnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(1.0, y.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ(1.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, y.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(3.0, y.xp.storage.base.data.real_double[2]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
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
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.storage.base.data.complex_single[2][1]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
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
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(
            &y, mtxvector_base, size, ynnz, yidx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_copy(&y, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(12, y.size);
        TEST_ASSERT_EQ(12, y.xp.storage.base.size);
        TEST_ASSERT_EQ(5, y.num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 0);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.num_nonzeros);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[0], 5);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[1], 6);
            TEST_ASSERT_EQ(y.xp.storage.base.idx[2], 9);
        }
        TEST_ASSERT_EQ(mtxvector_base, y.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, y.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, y.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, y.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0f, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-1.0f, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 2.0f, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(-2.0f, y.xp.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 3.0f, y.xp.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-3.0f, y.xp.storage.base.data.complex_double[2][1]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_scal()’ tests scaling vectors by a constant.
 */
int test_mtxvector_dist_scal(void)
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
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.xp.storage.base.data.real_single[2]);
        }
        err = mtxvector_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.xp.storage.base.data.real_single[2]);
        }
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(6.0f, x.xp.storage.base.data.real_double[2]);
        }
        err = mtxvector_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(12.0f, x.xp.storage.base.data.real_double[2]);
        }
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(4.0f, x.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(0.0f, x.xp.storage.base.data.complex_single[0][1]);
        }
        err = mtxvector_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.storage.base.data.complex_single[0][1]);
        }
        float as[2] = {2, 3};
        err = mtxvector_dist_cscal(as, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -4.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 20.0f, x.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(-16.0f, x.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 28.0f, x.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 24.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 36.0f, x.xp.storage.base.data.complex_single[0][1]);
        }
        double ad[2] = {2, 3};
        err = mtxvector_dist_zscal(ad, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -68.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(  28.0f, x.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(-116.0f, x.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(   8.0f, x.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( -60.0f, x.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 144.0f, x.xp.storage.base.data.complex_single[0][1]);
        }
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(2.0f, x.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(4.0f, x.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(0.0f, x.xp.storage.base.data.complex_double[0][1]);
        }
        err = mtxvector_dist_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 4.0f, x.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 8.0f, x.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0f, x.xp.storage.base.data.complex_double[0][1]);
        }
        float as[2] = {2, 3};
        err = mtxvector_dist_cscal(as, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -4.0, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 20.0, x.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(-16.0, x.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 28.0, x.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 24.0, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 36.0, x.xp.storage.base.data.complex_double[0][1]);
        }
        double ad[2] = {2, 3};
        err = mtxvector_dist_zscal(ad, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( -68.0, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(  28.0, x.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(-116.0, x.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(   8.0, x.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( -60.0, x.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 144.0, x.xp.storage.base.data.complex_double[0][1]);
        }
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_axpy()’ tests multiplying a vector by a
 * constant and adding the result to another vector.
 */
int test_mtxvector_dist_axpy(void)
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
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        float * ydata = rank == 0 ? (float[2]) {2.0f, 1.0f} : (float[3]) {0.0f, 2.0f, 1.0f};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 3.0f, y.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 2.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 6.0f, y.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ( 7.0f, y.xp.storage.base.data.real_single[2]);
        }
        err = mtxvector_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 5.0f, y.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.xp.storage.base.data.real_single[2]);
        }
        err = mtxvector_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 9.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.xp.storage.base.data.real_single[2]);
        }
        err = mtxvector_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.xp.storage.base.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.xp.storage.base.data.real_single[2]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        double * ydata = rank == 0 ? (double[2]) {2.0, 1.0} : (double[3]) {0.0, 2.0, 1.0};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 3.0, y.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 2.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 6.0, y.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ( 7.0, y.xp.storage.base.data.real_double[2]);
        }
        err = mtxvector_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 5.0, y.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 4.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.xp.storage.base.data.real_double[2]);
        }
        err = mtxvector_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 9.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.xp.storage.base.data.real_double[2]);
        }
        err = mtxvector_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.xp.storage.base.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.xp.storage.base.data.real_double[2]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{2.0f,1.0f}, {0.0f,2.0f}})
            : ((float[1][2]) {{1.0f,0.0f}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(2.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(6.0f, y.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(0.0f, y.xp.storage.base.data.complex_single[0][1]);
        }
        err = mtxvector_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 5.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 4.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(10.0f, y.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.xp.storage.base.data.complex_single[0][1]);
        }
        err = mtxvector_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 9.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(22.0f, y.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.xp.storage.base.data.complex_single[0][1]);
        }
        err = mtxvector_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(19.0f, y.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ(46.0f, y.xp.storage.base.data.complex_single[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0f, y.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ( 0.0f, y.xp.storage.base.data.complex_single[0][1]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0,1.0}, {1.0,2.0}})
            : ((double[1][2]) {{3.0,0.0}});
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{2.0,1.0}, {0.0,2.0}})
            : ((double[1][2]) {{1.0,0.0}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(2.0, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(6.0, y.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(0.0, y.xp.storage.base.data.complex_double[0][1]);
        }
        err = mtxvector_dist_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 5.0, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 4.0, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(10.0, y.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.xp.storage.base.data.complex_double[0][1]);
        }
        err = mtxvector_dist_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 9.0, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(22.0, y.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.xp.storage.base.data.complex_double[0][1]);
        }
        err = mtxvector_dist_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(19.0, y.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ(46.0, y.xp.storage.base.data.complex_double[1][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0, y.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ( 0.0, y.xp.storage.base.data.complex_double[0][1]);
        }
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_dot()’ tests computing the dot products of
 * pairs of vectors.
 */
int test_mtxvector_dist_dot(void)
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
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        float * ydata = rank == 0 ? (float[2]) {3.0f, 2.0f} : (float[3]) {1.0f, 0.0f, 1.0f};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_dist_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_dist_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        double * ydata = rank == 0 ? (double[2]) {3.0, 2.0} : (double[3]) {1.0, 0.0, 1.0};
        int nnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sdot;
        err = mtxvector_dist_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxvector_dist_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxvector_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        float (* ydata)[2] = rank == 0
            ? ((float[2][2]) {{3.0f,2.0f}, {1.0f,0.0f}})
            : ((float[1][2]) {{1.0f,0.0f}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist y;
        int size = 12;
        int64_t * idx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[1]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0,1.0}, {1.0,2.0}})
            : ((double[1][2]) {{3.0,0.0}});
        double (* ydata)[2] = rank == 0
            ? ((double[2][2]) {{3.0,2.0}, {1.0,0.0}})
            : ((double[1][2]) {{1.0,0.0}});
        int nnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, nnz, idx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(
            &y, mtxvector_base, size, nnz, idx, ydata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxvector_dist_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxvector_dist_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxvector_dist_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxvector_dist_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_nrm2()’ tests computing the Euclidean norm of
 * vectors.
 */
int test_mtxvector_dist_nrm2(void)
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
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{1.0f,1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float snrm2;
        err = mtxvector_dist_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxvector_dist_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_asum()’ tests computing the sum of
 * absolute values of vectors.
 */
int test_mtxvector_dist_asum(void)
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
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {-1.0f, 1.0f} : (float[3]) {1.0f, 2.0f, 3.0f};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {-1.0, 1.0} : (double[3]) {1.0, 2.0, 3.0};
        int xnnz = rank == 0 ? 2 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        float (* xdata)[2] = rank == 0
            ? ((float[2][2]) {{-1.0f,-1.0f}, {1.0f,2.0f}})
            : ((float[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5};
        double (* xdata)[2] = rank == 0
            ? ((double[2][2]) {{-1.0f,-1.0f}, {1.0f,2.0f}})
            : ((double[1][2]) {{3.0f,0.0f}});
        int xnnz = rank == 0 ? 2 : 1;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        float sasum;
        err = mtxvector_dist_sasum(&x, &sasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0f, sasum);
        double dasum;
        err = mtxvector_dist_dasum(&x, &dasum, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(8.0, dasum);
        mtxvector_dist_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxvector_dist_usscga()’ tests combined scatter-gather
 * operations from a distributed sparse vector in packed form to
 * another distributed sparse vector in packed form.
 */
int test_mtxvector_dist_usscga(void)
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
        struct mtxvector_dist x;
        struct mtxvector_dist z;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float * xdata = rank == 0 ? (float[2]) {1, 4} : (float[3]) {6, 7, 10};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * zidx = rank == 0 ? (int64_t[5]) {9, 0, 1, 6, 8} : (int64_t[3]) {3, 2, 5};
        float * zdata = rank == 0 ? (float[5]) {0, 0, 0, 0, 0} : (float[3]) {0, 0, 0};
        int znnz = rank == 0 ? 5 : 3;
        err = mtxvector_dist_init_real_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_single(
            &z, mtxvector_base, size, znnz, zidx, zdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_usscga(&z, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(mtxvector_base, z.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, z.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, z.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(5, z.xp.storage.base.size);
            TEST_ASSERT_EQ(10.0f, z.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 1.0f, z.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ( 0.0f, z.xp.storage.base.data.real_single[2]);
            TEST_ASSERT_EQ( 7.0f, z.xp.storage.base.data.real_single[3]);
            TEST_ASSERT_EQ( 0.0f, z.xp.storage.base.data.real_single[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 4.0f, z.xp.storage.base.data.real_single[0]);
            TEST_ASSERT_EQ( 0.0f, z.xp.storage.base.data.real_single[1]);
            TEST_ASSERT_EQ( 6.0f, z.xp.storage.base.data.real_single[2]);
        }
        mtxvector_dist_free(&z);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist z;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double * xdata = rank == 0 ? (double[2]) {1, 4} : (double[3]) {6, 7, 10};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * zidx = rank == 0 ? (int64_t[5]) {0, 1, 6, 8, 9} : (int64_t[3]) {3, 2, 5};
        double * zdata = rank == 0 ? (double[5]) {0, 0, 0, 0, 0} : (double[3]) {0, 0, 0};
        int znnz = rank == 0 ? 5 : 3;
        err = mtxvector_dist_init_real_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_real_double(
            &z, mtxvector_base, size, znnz, zidx, zdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_usscga(&z, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_base, z.xp.type);
        TEST_ASSERT_EQ(mtx_field_real, z.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, z.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(5, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 1.0, z.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 0.0, z.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ( 7.0, z.xp.storage.base.data.real_double[2]);
            TEST_ASSERT_EQ( 0.0, z.xp.storage.base.data.real_double[3]);
            TEST_ASSERT_EQ(10.0, z.xp.storage.base.data.real_double[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 4.0, z.xp.storage.base.data.real_double[0]);
            TEST_ASSERT_EQ( 0.0, z.xp.storage.base.data.real_double[1]);
            TEST_ASSERT_EQ( 6.0, z.xp.storage.base.data.real_double[2]);
        }
        mtxvector_dist_free(&z);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist z;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        float (* xdata)[2] = rank == 0 ? (float[2][2]) {{1,-1}, {4,-4}} : (float[3][2]) {{6,-6}, {7,-7}, {10,-10}};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * zidx = rank == 0 ? (int64_t[5]) {9, 0, 1, 6, 8} : (int64_t[3]) {3, 2, 5};
        float (* zdata)[2] = rank == 0 ? (float[5][2]) {{0,0}, {0,0}, {0,0}, {0,0}, {0,0}} : (float[3][2]) {{0,0}, {0,0}, {0,0}};
        int znnz = rank == 0 ? 5 : 3;
        err = mtxvector_dist_init_complex_single(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_single(
            &z, mtxvector_base, size, znnz, zidx, zdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_usscga(&z, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(mtxvector_base, z.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, z.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_single, z.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(5, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 10.0f, z.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-10.0f, z.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ(  1.0f, z.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ( -1.0f, z.xp.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ(  0.0f, z.xp.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(  0.0f, z.xp.storage.base.data.complex_single[2][1]);
            TEST_ASSERT_EQ(  7.0f, z.xp.storage.base.data.complex_single[3][0]);
            TEST_ASSERT_EQ( -7.0f, z.xp.storage.base.data.complex_single[3][1]);
            TEST_ASSERT_EQ(  0.0f, z.xp.storage.base.data.complex_single[4][0]);
            TEST_ASSERT_EQ(  0.0f, z.xp.storage.base.data.complex_single[4][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 4.0f, z.xp.storage.base.data.complex_single[0][0]);
            TEST_ASSERT_EQ(-4.0f, z.xp.storage.base.data.complex_single[0][1]);
            TEST_ASSERT_EQ( 0.0f, z.xp.storage.base.data.complex_single[1][0]);
            TEST_ASSERT_EQ( 0.0f, z.xp.storage.base.data.complex_single[1][1]);
            TEST_ASSERT_EQ( 6.0f, z.xp.storage.base.data.complex_single[2][0]);
            TEST_ASSERT_EQ(-6.0f, z.xp.storage.base.data.complex_single[2][1]);
        }
        mtxvector_dist_free(&z);
        mtxvector_dist_free(&x);
    }
    {
        struct mtxvector_dist x;
        struct mtxvector_dist z;
        int size = 12;
        int64_t * xidx = rank == 0 ? (int64_t[2]) {0, 3} : (int64_t[3]) {5, 6, 9};
        double (* xdata)[2] = rank == 0 ? (double[2][2]) {{1,-1}, {4,-4}} : (double[3][2]) {{6,-6}, {7,-7}, {10,-10}};
        int xnnz = rank == 0 ? 2 : 3;
        int64_t * zidx = rank == 0 ? (int64_t[5]) {9, 0, 1, 6, 8} : (int64_t[3]) {3, 2, 5};
        double (* zdata)[2] = rank == 0 ? (double[5][2]) {{0,0}, {0,0}, {0,0}, {0,0}, {0,0}} : (double[3][2]) {{0,0}, {0,0}, {0,0}};
        int znnz = rank == 0 ? 5 : 3;
        err = mtxvector_dist_init_complex_double(
            &x, mtxvector_base, size, xnnz, xidx, xdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_init_complex_double(
            &z, mtxvector_base, size, znnz, zidx, zdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxvector_dist_usscga(&z, &x, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxdisterror_description(&disterr));
        TEST_ASSERT_EQ(mtxvector_base, z.xp.type);
        TEST_ASSERT_EQ(mtx_field_complex, z.xp.storage.base.field);
        TEST_ASSERT_EQ(mtx_double, z.xp.storage.base.precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(5, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 10.0, z.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-10.0, z.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ(  1.0, z.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ( -1.0, z.xp.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ(  0.0, z.xp.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(  0.0, z.xp.storage.base.data.complex_double[2][1]);
            TEST_ASSERT_EQ(  7.0, z.xp.storage.base.data.complex_double[3][0]);
            TEST_ASSERT_EQ( -7.0, z.xp.storage.base.data.complex_double[3][1]);
            TEST_ASSERT_EQ(  0.0, z.xp.storage.base.data.complex_double[4][0]);
            TEST_ASSERT_EQ(  0.0, z.xp.storage.base.data.complex_double[4][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, z.xp.storage.base.size);
            TEST_ASSERT_EQ( 4.0, z.xp.storage.base.data.complex_double[0][0]);
            TEST_ASSERT_EQ(-4.0, z.xp.storage.base.data.complex_double[0][1]);
            TEST_ASSERT_EQ( 0.0, z.xp.storage.base.data.complex_double[1][0]);
            TEST_ASSERT_EQ( 0.0, z.xp.storage.base.data.complex_double[1][1]);
            TEST_ASSERT_EQ( 6.0, z.xp.storage.base.data.complex_double[2][0]);
            TEST_ASSERT_EQ(-6.0, z.xp.storage.base.data.complex_double[2][1]);
        }
        mtxvector_dist_free(&z);
        mtxvector_dist_free(&x);
    }
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
    TEST_SUITE_BEGIN("Running tests for distributed vectors\n");
    TEST_RUN(test_mtxvector_dist_from_mtxfile);
    TEST_RUN(test_mtxvector_dist_to_mtxfile);
    TEST_RUN(test_mtxvector_dist_from_mtxdistfile);
    TEST_RUN(test_mtxvector_dist_to_mtxdistfile);
    TEST_RUN(test_mtxvector_dist_split);
    TEST_RUN(test_mtxvector_dist_swap);
    TEST_RUN(test_mtxvector_dist_copy);
    TEST_RUN(test_mtxvector_dist_scal);
    TEST_RUN(test_mtxvector_dist_axpy);
    TEST_RUN(test_mtxvector_dist_dot);
    TEST_RUN(test_mtxvector_dist_nrm2);
    TEST_RUN(test_mtxvector_dist_asum);
    TEST_RUN(test_mtxvector_dist_usscga);
    TEST_SUITE_END();

    /* 3. clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
