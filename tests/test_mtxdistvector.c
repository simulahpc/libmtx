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
 * Last modified: 2022-01-19
 *
 * Unit tests for distributed Matrix Market files.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/distvector.h>
#include <libmtx/util/partition.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistvector";

/**
 * ‘test_mtxdistvector_from_mtxfile()’ tests converting Matrix Market
 * files stored on a single process to distributed vectors.
 */
int test_mtxdistvector_from_mtxfile(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = 8;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_vector_array_real_double(
            &srcmtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxfile(
            &mtxdistvector, &srcmtxfile, mtxvector_auto,
            NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_array, mtxdistvector.interior.type);
        const struct mtxvector_array * interior =
            &mtxdistvector.interior.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        TEST_ASSERT_EQ(4, interior->size);
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(5.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[3]);
        }
        mtxdistvector_free(&mtxdistvector);
        mtxfile_free(&srcmtxfile);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 7;
        const struct mtxfile_vector_coordinate_real_double mtxdata[] =
            {{1,1.0}, {2,2.0}, {3,3.0}, {5,5.0}, {6,6.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_vector_coordinate_real_double(
            &srcmtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxfile(
            &mtxdistvector, &srcmtxfile, mtxvector_auto,
            NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, mtxdistvector.interior.type);
        const struct mtxvector_coordinate * interior =
            &mtxdistvector.interior.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        TEST_ASSERT_EQ(rank == 0 ? 4 : 3, interior->size);
        TEST_ASSERT_EQ(rank == 0 ? 3 : 2, interior->num_nonzeros);
        TEST_ASSERT_EQ(7, mtxdistvector.rowpart.size);
        TEST_ASSERT_EQ(2, mtxdistvector.rowpart.num_parts);
        TEST_ASSERT_EQ(4, mtxdistvector.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(3, mtxdistvector.rowpart.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(0, interior->indices[0]);
            TEST_ASSERT_EQ(1, interior->indices[1]);
            TEST_ASSERT_EQ(2, interior->indices[2]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0, interior->indices[0]);
            TEST_ASSERT_EQ(1, interior->indices[1]);
            TEST_ASSERT_EQ(5.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[1]);
        }
        mtxdistvector_free(&mtxdistvector);
        mtxfile_free(&srcmtxfile);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_to_mtxfile()’ tests converting distributed
 * vectors to Matrix Market files stored on a single process.
 */
int test_mtxdistvector_to_mtxfile(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = 8;
        int num_local_rows = rank == 0 ? 2 : 6;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[6]) {3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,6};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector src;
        err = mtxdistvector_init_array_real_double(
            &src, num_local_rows, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxfile dst;
        err = mtxdistvector_to_mtxfile(
            &dst, &src, mtxfile_array, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(8, dst.size.num_rows);
            TEST_ASSERT_EQ(-1, dst.size.num_columns);
            TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
            const double * data = dst.data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
            TEST_ASSERT_EQ(6.0, data[5]);
            TEST_ASSERT_EQ(7.0, data[6]);
            TEST_ASSERT_EQ(8.0, data[7]);
            mtxfile_free(&dst);
        }
        mtxdistvector_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 9;
        int num_local_rows = rank == 0 ? 3 : 6;
        const int * srcidx = (rank == 0)
            ? ((const int[3]) {0, 1, 2})
            : ((const int[5]) {0, 1, 2, 3, 4});
        const double * srcdata = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 3.0})
            : ((const double[5]) {4.0, 5.0, 6.0, 7.0, 8.0});
        int64_t num_local_nonzeros = (rank == 0) ? 3 : 5;

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,6};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector src;
        err = mtxdistvector_init_coordinate_real_double(
            &src, num_local_rows, num_local_nonzeros, srcidx, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxfile dst;
        err = mtxdistvector_to_mtxfile(
            &dst, &src, mtxfile_coordinate, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(9, dst.size.num_rows);
            TEST_ASSERT_EQ(-1, dst.size.num_columns);
            TEST_ASSERT_EQ(8, dst.size.num_nonzeros);
            const struct mtxfile_vector_coordinate_real_double * data =
                dst.data.vector_coordinate_real_double;
            TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(  2, data[1].i); TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(  3, data[2].i); TEST_ASSERT_EQ(3.0, data[2].a);
            TEST_ASSERT_EQ(  4, data[3].i); TEST_ASSERT_EQ(4.0, data[3].a);
            TEST_ASSERT_EQ(  5, data[4].i); TEST_ASSERT_EQ(5.0, data[4].a);
            TEST_ASSERT_EQ(  6, data[5].i); TEST_ASSERT_EQ(6.0, data[5].a);
            TEST_ASSERT_EQ(  7, data[6].i); TEST_ASSERT_EQ(7.0, data[6].a);
            TEST_ASSERT_EQ(  8, data[7].i); TEST_ASSERT_EQ(8.0, data[7].a);
            mtxfile_free(&dst);
        }
        mtxdistvector_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_from_mtxdistfile()’ tests converting
 * distributed Matrix Market files to distributed vectors.
 */
int test_mtxdistvector_from_mtxdistfile(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = 8;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[6]) {3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,6};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_array_real_double(
            &src, num_rows, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxdistfile(
            &mtxdistvector, &src, mtxvector_auto,
            NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(8, mtxdistvector.rowpart.size);
        TEST_ASSERT_EQ(4, mtxdistvector.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(4, mtxdistvector.rowpart.part_sizes[1]);
        TEST_ASSERT_EQ(mtxvector_array, mtxdistvector.interior.type);
        const struct mtxvector_array * interior =
            &mtxdistvector.interior.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        TEST_ASSERT_EQ(4, interior->size);
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(5.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[3]);
        }
        mtxdistvector_free(&mtxdistvector);
        mtxdistfile_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 9;
        const struct mtxfile_vector_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_double[2])
                {{1,1.0}, {2,2.0}})
            : ((const struct mtxfile_vector_coordinate_real_double[6])
                {{3,3.0}, {4,4.0}, {5,5.0}, {6,6.0}, {7,7.0}, {8,8.0}});
        int64_t num_nonzeros = 8;

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,6};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxdistfile(
            &mtxdistvector, &src, mtxvector_auto,
            NULL, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9, mtxdistvector.rowpart.size);
        TEST_ASSERT_EQ(5, mtxdistvector.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(4, mtxdistvector.rowpart.part_sizes[1]);
        TEST_ASSERT_EQ(mtxvector_coordinate, mtxdistvector.interior.type);
        const struct mtxvector_coordinate * interior =
            &mtxdistvector.interior.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(5, interior->size);
            TEST_ASSERT_EQ(5, interior->num_nonzeros);
            TEST_ASSERT_EQ(0, interior->indices[0]);
            TEST_ASSERT_EQ(1, interior->indices[1]);
            TEST_ASSERT_EQ(2, interior->indices[2]);
            TEST_ASSERT_EQ(3, interior->indices[3]);
            TEST_ASSERT_EQ(4, interior->indices[4]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[3]);
            TEST_ASSERT_EQ(5.0, interior->data.real_double[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4, interior->size);
            TEST_ASSERT_EQ(3, interior->num_nonzeros);
            TEST_ASSERT_EQ(0, interior->indices[0]);
            TEST_ASSERT_EQ(1, interior->indices[1]);
            TEST_ASSERT_EQ(2, interior->indices[2]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[2]);
        }
        mtxdistvector_free(&mtxdistvector);
        mtxdistfile_free(&src);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_to_mtxdistfile()’ tests converting distributed
 * vectors to distributed Matrix Market files.
 */
int test_mtxdistvector_to_mtxdistfile(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = 8;
        int num_local_rows = rank == 0 ? 2 : 6;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[6]) {3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,6};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector src;
        err = mtxdistvector_init_array_real_double(
            &src, num_local_rows, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxdistfile dst;
        err = mtxdistvector_to_mtxdistfile(&dst, &src, mtxfile_array, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(8, dst.size.num_rows);
        TEST_ASSERT_EQ(-1, dst.size.num_columns);
        TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(8, dst.partition.size);
        TEST_ASSERT_EQ(2, dst.partition.num_parts);
        TEST_ASSERT_EQ(2, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(6, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3.0, data[0]);
            TEST_ASSERT_EQ(4.0, data[1]);
            TEST_ASSERT_EQ(5.0, data[2]);
            TEST_ASSERT_EQ(6.0, data[3]);
            TEST_ASSERT_EQ(7.0, data[4]);
            TEST_ASSERT_EQ(8.0, data[5]);
        }
        mtxdistfile_free(&dst);
        mtxdistvector_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 9;
        int num_local_rows = rank == 0 ? 3 : 6;
        const int * srcidx = (rank == 0)
            ? ((const int[3]) {0, 1, 2})
            : ((const int[5]) {0, 1, 2, 3, 4});
        const double * srcdata = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 3.0})
            : ((const double[5]) {4.0, 5.0, 6.0, 7.0, 8.0});
        int64_t num_local_nonzeros = (rank == 0) ? 3 : 5;

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,6};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector src;
        err = mtxdistvector_init_coordinate_real_double(
            &src, num_local_rows, num_local_nonzeros, srcidx, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxdistfile dst;
        err = mtxdistvector_to_mtxdistfile(&dst, &src, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(9, dst.size.num_rows);
        TEST_ASSERT_EQ(-1, dst.size.num_columns);
        TEST_ASSERT_EQ(8, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(8, dst.partition.size);
        TEST_ASSERT_EQ(2, dst.partition.num_parts);
        TEST_ASSERT_EQ(3, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[1]);
        const struct mtxfile_vector_coordinate_real_double * data =
            dst.data.vector_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(  2, data[1].i); TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(  3, data[2].i); TEST_ASSERT_EQ(3.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(  4, data[0].i); TEST_ASSERT_EQ(4.0, data[0].a);
            TEST_ASSERT_EQ(  5, data[1].i); TEST_ASSERT_EQ(5.0, data[1].a);
            TEST_ASSERT_EQ(  6, data[2].i); TEST_ASSERT_EQ(6.0, data[2].a);
            TEST_ASSERT_EQ(  7, data[3].i); TEST_ASSERT_EQ(7.0, data[3].a);
            TEST_ASSERT_EQ(  8, data[4].i); TEST_ASSERT_EQ(8.0, data[4].a);
        }
        mtxdistfile_free(&dst);
        mtxdistvector_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_dot()’ tests computing the dot products of
 * pairs of vectors.
 */
int test_mtxdistvector_dot(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistvector x;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_single(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        const float * ydata = (rank == 0)
            ? ((const float[2]) {3.0f, 2.0f})
            : ((const float[3]) {1.0f, 0.0f, 1.0f});
        err = mtxdistvector_init_array_real_single(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_double(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        const double * ydata = (rank == 0)
            ? ((const double[2]) {3.0, 2.0})
            : ((const double[3]) {1.0, 0.0, 1.0});
        err = mtxdistvector_init_array_real_double(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[][2]) {{1.0f, 1.0f}})
            : ((const float[][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_single(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[][2]) {{3.0f, 2.0f}})
            : ((const float[][2]) {{1.0f, 0.0f}, {1.0f, 0.0f}});
        err = mtxdistvector_init_array_complex_single(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[][2]) {{1.0, 1.0}})
            : ((const double[][2]) {{1.0, 2.0}, {3.0, 0.0}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_double(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[][2]) {{3.0, 2.0}})
            : ((const double[][2]) {{1.0, 0.0}, {1.0, 0.0}});
        err = mtxdistvector_init_array_complex_double(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const int32_t * xdata = (rank == 0)
            ? ((const int32_t[2]) {1, 1})
            : ((const int32_t[3]) {1, 2, 3});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_integer_single(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        const int32_t * ydata = (rank == 0)
            ? ((const int32_t[2]) {3, 2})
            : ((const int32_t[3]) {1, 0, 1});
        err = mtxdistvector_init_array_integer_single(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const int64_t * xdata = (rank == 0)
            ? ((const int64_t[2]) {1, 1})
            : ((const int64_t[3]) {1, 2, 3});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_integer_double(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        const int64_t * ydata = (rank == 0)
            ? ((const int64_t[2]) {3, 2})
            : ((const int64_t[3]) {1, 0, 1});
        err = mtxdistvector_init_array_integer_double(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1,3})
            : ((const int[3]) {0,2,4});
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        const float * ydata = (rank == 0)
            ? ((const float[2]) {3.0f, 2.0f})
            : ((const float[3]) {1.0f, 0.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_real_single(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        err = mtxdistvector_init_coordinate_real_single(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1,3})
            : ((const int[3]) {0,1,4});
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0f, 1.0f})
            : ((const double[3]) {1.0f, 2.0f, 3.0f});
        const double * ydata = (rank == 0)
            ? ((const double[2]) {3.0f, 2.0f})
            : ((const double[3]) {1.0f, 0.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_real_double(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        err = mtxdistvector_init_coordinate_real_double(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 1 : 2;
        const int * idx = (rank == 0)
            ? ((const int[1]) {1})
            : ((const int[2]) {3, 5});
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f, 1.0f}})
            : ((const float[2][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[1][2]) {{3.0f, 2.0f}})
            : ((const float[2][2]) {{1.0f, 0.0f}, {1.0f, 0.0f}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_complex_single(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        err = mtxdistvector_init_coordinate_complex_single(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 1 : 2;
        const int * idx = (rank == 0)
            ? ((const int[1]) {1})
            : ((const int[2]) {3, 5});
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0f, 1.0f}})
            : ((const double[2][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[1][2]) {{3.0f, 2.0f}})
            : ((const double[2][2]) {{1.0f, 0.0f}, {1.0f, 0.0f}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_complex_double(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        err = mtxdistvector_init_coordinate_complex_double(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1,3})
            : ((const int[3]) {0,1,4});
        const int32_t * xdata = (rank == 0)
            ? ((const int32_t[2]) {1.0f, 1.0f})
            : ((const int32_t[3]) {1.0f, 2.0f, 3.0f});
        const int32_t * ydata = (rank == 0)
            ? ((const int32_t[2]) {3.0f, 2.0f})
            : ((const int32_t[3]) {1.0f, 0.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_integer_single(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        err = mtxdistvector_init_coordinate_integer_single(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1,3})
            : ((const int[3]) {0,1,4});
        const int64_t * xdata = (rank == 0)
            ? ((const int64_t[2]) {1.0f, 1.0f})
            : ((const int64_t[3]) {1.0f, 2.0f, 3.0f});
        const int64_t * ydata = (rank == 0)
            ? ((const int64_t[2]) {3.0f, 2.0f})
            : ((const int64_t[3]) {1.0f, 0.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_integer_double(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        struct mtxdistvector y;
        err = mtxdistvector_init_coordinate_integer_double(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_nrm2()’ tests computing the Euclidean norm of
 * vectors.
 */
int test_mtxdistvector_nrm2(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistvector x;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const float * data = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_single(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(4.0f, snrm2, "snrm2=%f", snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_double(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const float (* data)[2] = (rank == 0)
            ? ((const float[][2]) {1.0f,1.0f})
            : ((const float[][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_single(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double (* data)[2] = (rank == 0)
            ? ((const double[][2]) {1.0,1.0})
            : ((const double[][2]) {{1.0,2.0}, {3.0,0.0}});
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_double(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const int32_t * data = (rank == 0)
            ? ((const int32_t[2]) {1, 1})
            : ((const int32_t[3]) {1, 2, 3});
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_integer_single(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const int64_t * data = (rank == 0)
            ? ((const int64_t[2]) {1, 1})
            : ((const int64_t[3]) {1, 2, 3});
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_integer_double(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
        mtxpartition_free(&partition);
    }

    /*
     * Coordinate formats
     */

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {0,1,4});
        const float * data = (rank == 0)
            ? ((const float[2]) {1.0f,1.0f})
            : ((const float[3]) {1.0f,2.0f,3.0f});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_real_single(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
        mtxpartition_free(&partition);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {0,1,4});
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0,1.0})
            : ((const double[3]) {1.0,2.0,3.0});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_real_double(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {0,4});
        const float (* data)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f,1.0f}})
            : ((const float[2][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_complex_single(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {0,4});
        const double (* data)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0,1.0}})
            : ((const double[2][2]) {{1.0,2.0}, {3.0,0.0}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_complex_double(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {0,1,4});
        const int32_t * data = (rank == 0)
            ? ((const int32_t[2]) {1,1})
            : ((const int32_t[3]) {1,2,3});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_integer_single(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {0,1,4});
        const int64_t * data = (rank == 0)
            ? ((const int64_t[2]) {1,1})
            : ((const int64_t[3]) {1,2,3});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_integer_double(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 24;
        int localsize = rank == 0 ? 9 : 15;
        const int * indices = (rank == 0)
            ? ((const int[7]) {0,2,3,4,5,6,7})
            : ((const int[9]) {0,1,2,4,5,8,11,12,14});
        size_t num_nonzeros = (rank == 0) ? 7 : 9;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {9,15};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_pattern(
            &x, localsize, num_nonzeros, indices, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_scal()’ tests scaling vectors by a constant.
 */
int test_mtxdistvector_scal(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistvector x;
        const float * data = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_single(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.array.data.real_single[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_double(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(6.0, x.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(8.0, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(12.0, x.interior.storage.array.data.real_double[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const float (* data)[2] = (rank == 0)
            ? ((const float[][2]) {1.0f,1.0f})
            : ((const float[][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_single(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.complex_single[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double (* data)[2] = (rank == 0)
            ? ((const double[][2]) {1.0,1.0})
            : ((const double[][2]) {{1.0,2.0}, {3.0,0.0}});
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_double(
            &x, localsize, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(6.0, x.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(8.0, x.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(12.0, x.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.complex_double[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {0,1,4});
        const float * data = (rank == 0)
            ? ((const float[2]) {1.0f,1.0f})
            : ((const float[3]) {1.0f,2.0f,3.0f});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_real_single(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.real_single[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {0,1,4});
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0,1.0})
            : ((const double[3]) {1.0,2.0,3.0});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_real_double(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.real_double[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {0,4});
        const float (* data)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f,1.0f}})
            : ((const float[2][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_complex_single(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_single[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {0,4});
        const double (* data)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0,1.0}})
            : ((const double[2][2]) {{1.0,2.0}, {3.0,0.0}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistvector_init_coordinate_complex_double(
            &x, localsize, num_nonzeros, indices, data, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_double[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&x);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistvector_axpy()’ tests multiplying a vector by a constant
 * and adding the result to another vector.
 */
int test_mtxdistvector_axpy(void)
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
    if (comm_size != 2)
        TEST_FAIL_MSG("Expected exactly two MPI processes");

    struct mtxdisterror disterr;
    err = mtxdisterror_alloc(&disterr, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        const float * ydata = (rank == 0)
            ? ((const float[2]) {2.0f, 1.0f})
            : ((const float[3]) {0.0f, 2.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_single(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_array_real_single(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.array.data.real_single[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 5;
        int localsize = rank == 0 ? 2 : 3;
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});
        const double * ydata = (rank == 0)
            ? ((const double[2]) {2.0, 1.0})
            : ((const double[3]) {0.0, 2.0, 1.0});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_real_double(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_array_real_double(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.array.data.real_double[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[][2]) {{1.0f, 1.0f}})
            : ((const float[][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[][2]) {{2.0f, 1.0f}})
            : ((const float[][2]) {{0.0f, 2.0f}, {1.0f, 0.0f}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_single(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_array_complex_single(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 3;
        int localsize = rank == 0 ? 1 : 2;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[][2]) {{1.0, 1.0}})
            : ((const double[][2]) {{1.0, 2.0}, {3.0, 0.0}});
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[][2]) {{2.0, 1.0}})
            : ((const double[][2]) {{0.0, 2.0}, {1.0, 0.0}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {1,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_array_complex_double(
            &x, localsize, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_array_complex_double(
            &y, localsize, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1,3})
            : ((const int[3]) {0,1,4});
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        const float * ydata = (rank == 0)
            ? ((const float[2]) {2.0f, 1.0f})
            : ((const float[3]) {0.0f, 2.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_coordinate_real_single(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_coordinate_real_single(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1,3})
            : ((const int[3]) {0,1,4});
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0f, 1.0f})
            : ((const double[3]) {1.0f, 2.0f, 3.0f});
        const double * ydata = (rank == 0)
            ? ((const double[2]) {2.0f, 1.0f})
            : ((const double[3]) {0.0f, 2.0f, 1.0f});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_coordinate_real_double(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_coordinate_real_double(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 1 : 2;
        const int * idx = (rank == 0)
            ? ((const int[1]) {1})
            : ((const int[2]) {3, 5});
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f, 1.0f}})
            : ((const float[2][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[1][2]) {{2.0f, 1.0f}})
            : ((const float[2][2]) {{0.0f, 2.0f}, {1.0f, 0.0f}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_coordinate_complex_single(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_coordinate_complex_single(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
        int localsize = rank == 0 ? 5 : 7;
        int nnz = (rank == 0) ? 1 : 2;
        const int * idx = (rank == 0)
            ? ((const int[1]) {1})
            : ((const int[2]) {3, 5});
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0f, 1.0f}})
            : ((const double[2][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[1][2]) {{2.0f, 1.0f}})
            : ((const double[2][2]) {{0.0f, 2.0f}, {1.0f, 0.0f}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {5,7};
        struct mtxpartition partition;
        err = mtxpartition_init_block(&partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        err = mtxdistvector_init_coordinate_complex_double(
            &x, localsize, nnz, idx, xdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_init_coordinate_complex_double(
            &y, localsize, nnz, idx, ydata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        mtxpartition_free(&partition);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
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

    /* 1. Initialise MPI. */
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const int mpi_root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for distributed vectors\n");
    TEST_RUN(test_mtxdistvector_from_mtxfile);
    TEST_RUN(test_mtxdistvector_to_mtxfile);
    TEST_RUN(test_mtxdistvector_from_mtxdistfile);
    TEST_RUN(test_mtxdistvector_to_mtxdistfile);
    TEST_RUN(test_mtxdistvector_dot);
    TEST_RUN(test_mtxdistvector_nrm2);
    TEST_RUN(test_mtxdistvector_scal);
    TEST_RUN(test_mtxdistvector_axpy);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
