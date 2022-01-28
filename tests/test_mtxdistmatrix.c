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
 * Last modified: 2022-01-26
 *
 * Unit tests for distributed Matrix Market files.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/distmatrix.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/distvector.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistmatrix";

/**
 * ‘test_mtxdistmatrix_from_mtxfile()’ tests converting Matrix Market
 * files stored on a single process to distributed matrices.
 */
int test_mtxdistmatrix_from_mtxfile(void)
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
        int num_rows = 3;
        int num_columns = 3;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_matrix_array_real_double(
            &srcmtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistmatrix mtxdistmatrix;
        err = mtxdistmatrix_from_mtxfile(
            &mtxdistmatrix, &srcmtxfile, mtxmatrix_auto,
            NULL, NULL, comm, root, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_array, mtxdistmatrix.interior.type);
        const struct mtxmatrix_array * interior =
            &mtxdistmatrix.interior.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, interior->num_rows);
            TEST_ASSERT_EQ(3, interior->num_columns);
            TEST_ASSERT_EQ(6, interior->size);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[3]);
            TEST_ASSERT_EQ(5.0, interior->data.real_double[4]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, interior->num_rows);
            TEST_ASSERT_EQ(3, interior->num_columns);
            TEST_ASSERT_EQ(3, interior->size);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(9.0, interior->data.real_double[2]);
        }
        mtxdistmatrix_free(&mtxdistmatrix);
        mtxfile_free(&srcmtxfile);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double mtxdata[] =
            {{1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,1,7.0}, {3,3,9.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistmatrix mtxdistmatrix;
        err = mtxdistmatrix_from_mtxfile(
            &mtxdistmatrix, &srcmtxfile, mtxmatrix_auto,
            NULL, NULL, comm, root, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, mtxdistmatrix.interior.type);
        const struct mtxmatrix_coordinate * interior =
            &mtxdistmatrix.interior.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        TEST_ASSERT_EQ(3, mtxdistmatrix.rowpart.size);
        TEST_ASSERT_EQ(2, mtxdistmatrix.rowpart.num_parts);
        TEST_ASSERT_EQ(2, mtxdistmatrix.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistmatrix.rowpart.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, interior->num_rows);
            TEST_ASSERT_EQ(3, interior->num_columns);
            TEST_ASSERT_EQ(3, interior->num_nonzeros);
            TEST_ASSERT_EQ(0, interior->rowidx[0]);
            TEST_ASSERT_EQ(0, interior->rowidx[1]);
            TEST_ASSERT_EQ(1, interior->rowidx[2]);
            TEST_ASSERT_EQ(0, interior->colidx[0]);
            TEST_ASSERT_EQ(2, interior->colidx[1]);
            TEST_ASSERT_EQ(0, interior->colidx[2]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, interior->num_rows);
            TEST_ASSERT_EQ(3, interior->num_columns);
            TEST_ASSERT_EQ(2, interior->num_nonzeros);
            TEST_ASSERT_EQ(0, interior->rowidx[0]);
            TEST_ASSERT_EQ(0, interior->rowidx[1]);
            TEST_ASSERT_EQ(0, interior->colidx[0]);
            TEST_ASSERT_EQ(2, interior->colidx[1]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(9.0, interior->data.real_double[1]);
        }
        mtxdistmatrix_free(&mtxdistmatrix);
        mtxfile_free(&srcmtxfile);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double mtxdata[] =
            {{1,1,1.0}, {1,3,3.0}, {2,1,4.0}, {3,1,7.0}, {3,3,9.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxfile, mtxfile_general, num_rows, num_columns,
            num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxpartition colpart;
        err = mtxpartition_init_block(&colpart, num_columns, comm_size, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistmatrix mtxdistmatrix;
        err = mtxdistmatrix_from_mtxfile(
            &mtxdistmatrix, &srcmtxfile, mtxmatrix_auto,
            NULL, &colpart, comm, root, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxmatrix_coordinate, mtxdistmatrix.interior.type);
        const struct mtxmatrix_coordinate * interior =
            &mtxdistmatrix.interior.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(3, interior->num_rows);
            TEST_ASSERT_EQ(2, interior->num_columns);
            TEST_ASSERT_EQ(3, interior->num_nonzeros);
            TEST_ASSERT_EQ(0, interior->rowidx[0]);
            TEST_ASSERT_EQ(0, interior->colidx[0]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(1, interior->rowidx[1]);
            TEST_ASSERT_EQ(0, interior->colidx[1]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(2, interior->rowidx[2]);
            TEST_ASSERT_EQ(0, interior->colidx[2]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, interior->num_rows);
            TEST_ASSERT_EQ(1, interior->num_columns);
            TEST_ASSERT_EQ(2, interior->num_nonzeros);
            TEST_ASSERT_EQ(0, interior->rowidx[0]);
            TEST_ASSERT_EQ(0, interior->colidx[0]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2, interior->rowidx[1]);
            TEST_ASSERT_EQ(0, interior->colidx[1]);
            TEST_ASSERT_EQ(9.0, interior->data.real_double[1]);
        }
        mtxdistmatrix_free(&mtxdistmatrix);
        mtxpartition_free(&colpart);
        mtxfile_free(&srcmtxfile);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_to_mtxfile()’ tests converting distributed
 * matrices to Matrix Market files stored on a single process.
 */
int test_mtxdistmatrix_to_mtxfile(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistmatrix src;
        err = mtxdistmatrix_init_array_real_double(
            &src, num_local_rows, num_local_columns,
            srcdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxdistmatrix_to_mtxfile(
            &dst, &src, mtxfile_array, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(3, dst.size.num_rows);
            TEST_ASSERT_EQ(3, dst.size.num_columns);
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
            TEST_ASSERT_EQ(9.0, data[8]);
            mtxfile_free(&dst);
        }
        mtxdistmatrix_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const int * rowidx = (rank == 0)
            ? ((const int[3]) {0, 0, 1})
            : ((const int[2]) {0, 0});
        const int * colidx = (rank == 0)
            ? ((const int[3]) {0, 1, 2})
            : ((const int[2]) {0, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 6.0})
            : ((const double[2]) {7.0, 9.0});
        int64_t num_local_nonzeros = (rank == 0) ? 3 : 2;

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,1};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistmatrix src;
        err = mtxdistmatrix_init_coordinate_real_double(
            &src, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, srcdata, &partition, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxfile dst;
        err = mtxdistmatrix_to_mtxfile(
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
        mtxdistmatrix_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_from_mtxdistfile()’ tests converting
 * distributed Matrix Market files to distributed matrices.
 */
int test_mtxdistmatrix_from_mtxdistfile(void)
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
        int num_rows = 3;
        int num_columns = 3;
        int num_nonzeros = 9;
        const double * srcdata = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 3.0, 4.0})
            : ((const double[5]) {5.0, 6.0, 7.0, 8.0, 9.0});

        int num_parts = comm_size;
        int64_t part_sizes[] = {4,5};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxdistmatrix mtxdistmatrix;
        err = mtxdistmatrix_from_mtxdistfile(
            &mtxdistmatrix, &src, mtxmatrix_auto,
            NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(3, mtxdistmatrix.rowpart.size);
        TEST_ASSERT_EQ(2, mtxdistmatrix.rowpart.num_parts);
        TEST_ASSERT_EQ(2, mtxdistmatrix.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistmatrix.rowpart.part_sizes[1]);
        TEST_ASSERT_EQ(3, mtxdistmatrix.colpart.size);
        TEST_ASSERT_EQ(1, mtxdistmatrix.colpart.num_parts);
        TEST_ASSERT_EQ(3, mtxdistmatrix.colpart.part_sizes[0]);
        TEST_ASSERT_EQ(mtxmatrix_array, mtxdistmatrix.interior.type);
        const struct mtxmatrix_array * interior =
            &mtxdistmatrix.interior.storage.array;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, interior->num_rows);
            TEST_ASSERT_EQ(3, interior->num_columns);
            TEST_ASSERT_EQ(6, interior->size);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[3]);
            TEST_ASSERT_EQ(5.0, interior->data.real_double[4]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(9.0, interior->data.real_double[2]);
        }
        mtxdistmatrix_free(&mtxdistmatrix);
        mtxdistfile_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t num_nonzeros = 7;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3])
                {{1,1,1.0}, {1,2,2.0}, {2,3,6.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[4])
                {{1,3,3.0}, {2,1,4.0}, {2,2,5.0}, {3,2,8.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,4};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros,
            srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&partition);

        struct mtxdistmatrix mtxdistmatrix;
        err = mtxdistmatrix_from_mtxdistfile(
            &mtxdistmatrix, &src, mtxmatrix_auto,
            NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(3, mtxdistmatrix.rowpart.size);
        TEST_ASSERT_EQ(2, mtxdistmatrix.rowpart.num_parts);
        TEST_ASSERT_EQ(2, mtxdistmatrix.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistmatrix.rowpart.part_sizes[1]);
        TEST_ASSERT_EQ(3, mtxdistmatrix.colpart.size);
        TEST_ASSERT_EQ(1, mtxdistmatrix.colpart.num_parts);
        TEST_ASSERT_EQ(3, mtxdistmatrix.colpart.part_sizes[0]);
        TEST_ASSERT_EQ(mtxmatrix_coordinate, mtxdistmatrix.interior.type);
        const struct mtxmatrix_coordinate * A =
            &mtxdistmatrix.interior.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, A->field);
        TEST_ASSERT_EQ(mtx_double, A->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(2, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(6, A->num_nonzeros);
            TEST_ASSERT_EQ(0, A->rowidx[0]); TEST_ASSERT_EQ(0, A->colidx[0]);
            TEST_ASSERT_EQ(0, A->rowidx[1]); TEST_ASSERT_EQ(1, A->colidx[1]);
            TEST_ASSERT_EQ(1, A->rowidx[2]); TEST_ASSERT_EQ(2, A->colidx[2]);
            TEST_ASSERT_EQ(0, A->rowidx[3]); TEST_ASSERT_EQ(2, A->colidx[3]);
            TEST_ASSERT_EQ(1, A->rowidx[4]); TEST_ASSERT_EQ(0, A->colidx[4]);
            TEST_ASSERT_EQ(1, A->rowidx[5]); TEST_ASSERT_EQ(1, A->colidx[5]);
            TEST_ASSERT_EQ(1.0, A->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, A->data.real_double[1]);
            TEST_ASSERT_EQ(6.0, A->data.real_double[2]);
            TEST_ASSERT_EQ(3.0, A->data.real_double[3]);
            TEST_ASSERT_EQ(4.0, A->data.real_double[4]);
            TEST_ASSERT_EQ(5.0, A->data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, A->num_rows);
            TEST_ASSERT_EQ(3, A->num_columns);
            TEST_ASSERT_EQ(1, A->num_nonzeros);
            TEST_ASSERT_EQ(0, A->rowidx[0]); TEST_ASSERT_EQ(1, A->colidx[0]);
            TEST_ASSERT_EQ(8.0, A->data.real_double[0]);
        }
        mtxdistmatrix_free(&mtxdistmatrix);
        mtxdistfile_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_to_mtxdistfile()’ tests converting distributed
 * matrices to distributed Matrix Market files.
 */
int test_mtxdistmatrix_to_mtxdistfile(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistmatrix src;
        err = mtxdistmatrix_init_array_real_double(
            &src, num_local_rows, num_local_columns,
            srcdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxdistfile dst;
        err = mtxdistmatrix_to_mtxdistfile(&dst, &src, mtxfile_array, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(3, dst.size.num_rows);
        TEST_ASSERT_EQ(3, dst.size.num_columns);
        TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(9, dst.partition.size);
        TEST_ASSERT_EQ(2, dst.partition.num_parts);
        TEST_ASSERT_EQ(6, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(3, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
            TEST_ASSERT_EQ(6.0, data[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(7.0, data[0]);
            TEST_ASSERT_EQ(8.0, data[1]);
            TEST_ASSERT_EQ(9.0, data[2]);
        }
        mtxdistfile_free(&dst);
        mtxdistmatrix_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const int * rowidx = (rank == 0)
            ? ((const int[3]) {0, 0, 1})
            : ((const int[2]) {0, 0});
        const int * colidx = (rank == 0)
            ? ((const int[3]) {0, 1, 2})
            : ((const int[2]) {0, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 6.0})
            : ((const double[2]) {7.0, 9.0});
        int64_t num_local_nonzeros = (rank == 0) ? 3 : 2;

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,1};
        struct mtxpartition rowpart;
        err = mtxpartition_init_block(
            &rowpart, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistmatrix src;
        err = mtxdistmatrix_init_coordinate_real_double(
            &src, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, srcdata, &rowpart, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&rowpart);

        struct mtxdistfile dst;
        err = mtxdistmatrix_to_mtxdistfile(&dst, &src, mtxfile_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(3, dst.size.num_rows);
        TEST_ASSERT_EQ(3, dst.size.num_columns);
        TEST_ASSERT_EQ(5, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(5, dst.partition.size);
        TEST_ASSERT_EQ(2, dst.partition.num_parts);
        TEST_ASSERT_EQ(3, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, dst.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            dst.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(  1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(  1, data[1].i); TEST_ASSERT_EQ(  2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(  2, data[2].i); TEST_ASSERT_EQ(  3, data[2].j);
            TEST_ASSERT_EQ(6.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(  3, data[0].i); TEST_ASSERT_EQ(  1, data[0].j);
            TEST_ASSERT_EQ(7.0, data[0].a);
            TEST_ASSERT_EQ(  3, data[1].i); TEST_ASSERT_EQ(  3, data[1].j);
            TEST_ASSERT_EQ(9.0, data[1].a);
        }
        mtxdistfile_free(&dst);
        mtxdistmatrix_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_alloc_row_vector()’ tests allocating row
 * vectors for distributed matrices.
 */
int test_mtxdistmatrix_alloc_row_vector(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistmatrix src;
        err = mtxdistmatrix_init_array_real_double(
            &src, num_local_rows, num_local_columns,
            srcdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxdistvector x;
        err = mtxdistmatrix_alloc_row_vector(
            &src, &x, mtxvector_array, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(1, x.comm_size);
        TEST_ASSERT_EQ(mtx_singleton, x.rowpart.type);
        TEST_ASSERT_EQ(3, x.rowpart.size);
        TEST_ASSERT_EQ(1, x.rowpart.num_parts);
        TEST_ASSERT_EQ(3, x.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(mtxvector_array, x.interior.type);
        TEST_ASSERT_EQ(mtx_field_real, x.interior.storage.array.field);
        TEST_ASSERT_EQ(mtx_double, x.interior.storage.array.precision);
        TEST_ASSERT_EQ(3, x.interior.storage.array.size);
        mtxdistvector_free(&x);

        struct mtxdistvector y;
        err = mtxdistmatrix_alloc_column_vector(
            &src, &y, mtxvector_array, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(2, y.comm_size);
        TEST_ASSERT_EQ(mtx_block, y.rowpart.type);
        TEST_ASSERT_EQ(3, y.rowpart.size);
        TEST_ASSERT_EQ(2, y.rowpart.num_parts);
        TEST_ASSERT_EQ(2, y.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(1, y.rowpart.part_sizes[1]);
        TEST_ASSERT_EQ(mtxvector_array, y.interior.type);
        TEST_ASSERT_EQ(mtx_field_real, y.interior.storage.array.field);
        TEST_ASSERT_EQ(mtx_double, y.interior.storage.array.precision);
        TEST_ASSERT_EQ(rank == 0 ? 2 : 1, y.interior.storage.array.size);
        mtxdistvector_free(&y);
        mtxdistmatrix_free(&src);
    }

    /*
     * Coordinate formats
     */
    {
        int num_rows = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const int * rowidx = (rank == 0)
            ? ((const int[3]) {0, 0, 1})
            : ((const int[2]) {0, 0});
        const int * colidx = (rank == 0)
            ? ((const int[3]) {0, 1, 2})
            : ((const int[2]) {0, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[3]) {1.0, 2.0, 6.0})
            : ((const double[2]) {7.0, 9.0});
        int64_t num_local_nonzeros = (rank == 0) ? 3 : 2;

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,1};
        struct mtxpartition rowpart;
        err = mtxpartition_init_block(
            &rowpart, num_rows, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistmatrix src;
        err = mtxdistmatrix_init_coordinate_real_double(
            &src, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, srcdata, &rowpart, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        mtxpartition_free(&rowpart);

        struct mtxdistvector x;
        err = mtxdistmatrix_alloc_row_vector(
            &src, &x, mtxvector_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(1, x.comm_size);
        TEST_ASSERT_EQ(mtx_singleton, x.rowpart.type);
        TEST_ASSERT_EQ(3, x.rowpart.size);
        TEST_ASSERT_EQ(1, x.rowpart.num_parts);
        TEST_ASSERT_EQ(3, x.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(mtxvector_coordinate, x.interior.type);
        TEST_ASSERT_EQ(mtx_field_real, x.interior.storage.array.field);
        TEST_ASSERT_EQ(mtx_double, x.interior.storage.array.precision);
        TEST_ASSERT_EQ(3, x.interior.storage.array.size);
        mtxdistvector_free(&x);

        struct mtxdistvector y;
        err = mtxdistmatrix_alloc_column_vector(
            &src, &y, mtxvector_coordinate, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(2, y.comm_size);
        TEST_ASSERT_EQ(mtx_block, y.rowpart.type);
        TEST_ASSERT_EQ(3, y.rowpart.size);
        TEST_ASSERT_EQ(2, y.rowpart.num_parts);
        TEST_ASSERT_EQ(2, y.rowpart.part_sizes[0]);
        TEST_ASSERT_EQ(1, y.rowpart.part_sizes[1]);
        TEST_ASSERT_EQ(mtxvector_coordinate, y.interior.type);
        TEST_ASSERT_EQ(mtx_field_real, y.interior.storage.array.field);
        TEST_ASSERT_EQ(mtx_double, y.interior.storage.array.precision);
        TEST_ASSERT_EQ(rank == 0 ? 2 : 1, y.interior.storage.array.size);
        mtxdistvector_free(&y);
        mtxdistmatrix_free(&src);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_dot()’ tests computing the dot products of
 * pairs of matrices.
 */
int test_mtxdistmatrix_dot(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * xdata = (rank == 0)
            ? ((const double[6]) {1.0, 1.0, 1.0, 0.0, 0.0, 2.0})
            : ((const double[3]) {0.0, 0.0, 3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_real_double(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        const double * ydata = (rank == 0)
            ? ((const double[6]) {3.0, 2.0, 1.0, 0.0, 0.0, 0.0})
            : ((const double[3]) {0.0, 0.0, 1.0});
        struct mtxdistmatrix y;
        err = mtxdistmatrix_init_array_real_double(
            &y, num_local_rows, num_local_columns,
            ydata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        float sdot;
        err = mtxdistmatrix_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistmatrix_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistmatrix_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistmatrix_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistmatrix_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistmatrix_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistmatrix_free(&y);
        mtxdistmatrix_free(&x);
    }

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[][2]) {
                    {1.0,1.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}})
            : ((const double[][2]) {{1.0,2.0}, {0.0,0.0}, {3.0,0.0}});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_complex_double(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[][2]) {
                    {3.0,2.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}})
            : ((const double[][2]) {{1.0,0.0}, {0.0,0.0}, {1.0,0.0}});
        struct mtxdistmatrix y;
        err = mtxdistmatrix_init_array_complex_double(
            &y, num_local_rows, num_local_columns,
            ydata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        float sdot;
        err = mtxdistmatrix_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        double ddot;
        err = mtxdistmatrix_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtxstrerror(err));
        float cdotu[2];
        err = mtxdistmatrix_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistmatrix_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistmatrix_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistmatrix_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxdistmatrix_free(&y);
        mtxdistmatrix_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        const int * rowidx = (rank == 0)
            ? ((const int[4]) {0, 0, 0, 1})
            : ((const int[1]) {0});
        const int * colidx = (rank == 0)
            ? ((const int[4]) {0, 1, 2, 2})
            : ((const int[1]) {2});
        const double * xdata = (rank == 0)
            ? ((const double[4]) {1.0, 1.0, 1.0, 2.0})
            : ((const double[1]) {3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_coordinate_real_double(
            &x, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        const double * ydata = (rank == 0)
            ? ((const double[4]) {3.0, 2.0, 1.0, 0.0})
            : ((const double[1]) {1.0});
        struct mtxdistmatrix y;
        err = mtxdistmatrix_init_coordinate_real_double(
            &y, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, ydata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        float sdot;
        err = mtxdistmatrix_sdot(&x, &y, &sdot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistmatrix_ddot(&x, &y, &ddot, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistmatrix_cdotu(&x, &y, &cdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistmatrix_cdotc(&x, &y, &cdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistmatrix_zdotu(&x, &y, &zdotu, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistmatrix_zdotc(&x, &y, &zdotc, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistmatrix_free(&y);
        mtxdistmatrix_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_nrm2()’ tests computing the Euclidean norm of
 * matrices.
 */
int test_mtxdistmatrix_nrm2(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * xdata = (rank == 0)
            ? ((const double[6]) {1.0, 1.0, 1.0, 0.0, 0.0, 2.0})
            : ((const double[3]) {0.0, 0.0, 3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_real_double(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        float snrm2;
        err = mtxdistmatrix_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(4.0f, snrm2, "snrm2=%f", snrm2);
        double dnrm2;
        err = mtxdistmatrix_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistmatrix_free(&x);
    }

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[][2]) {
                    {1.0,1.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {1.0,2.0}})
            : ((const double[][2]) {{0.0,0.0}, {0.0,0.0}, {3.0,0.0}});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_complex_double(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        float snrm2;
        err = mtxdistmatrix_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ(MTX_ERR_MPI_COLLECTIVE, err);
        TEST_ASSERT_EQ_MSG(MTX_ERR_INVALID_FIELD, disterr.err,
                           "%s", mtxstrerror(disterr.err));
        double dnrm2;
        err = mtxdistmatrix_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ(MTX_ERR_MPI_COLLECTIVE, err);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_FIELD, disterr.err);
        mtxdistmatrix_free(&x);
    }

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const int32_t * xdata = (rank == 0)
            ? ((const int32_t[6]) {1, 1, 1, 0, 0, 2})
            : ((const int32_t[3]) {0, 0, 3});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_integer_single(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistmatrix_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(4.0f, snrm2, "snrm2=%f", snrm2);
        double dnrm2;
        err = mtxdistmatrix_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistmatrix_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        const int * rowidx = (rank == 0)
            ? ((const int[4]) {0, 0, 0, 1})
            : ((const int[1]) {0});
        const int * colidx = (rank == 0)
            ? ((const int[4]) {0, 1, 2, 2})
            : ((const int[1]) {2});
        const double * xdata = (rank == 0)
            ? ((const double[4]) {1.0, 1.0, 1.0, 2.0})
            : ((const double[1]) {3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_coordinate_real_double(
            &x, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        float snrm2;
        err = mtxdistmatrix_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(4.0f, snrm2, "snrm2=%f", snrm2);
        double dnrm2;
        err = mtxdistmatrix_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistmatrix_free(&x);
    }

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        const int * rowidx = (rank == 0)
            ? ((const int[4]) {0, 0, 0, 1})
            : ((const int[1]) {0});
        const int * colidx = (rank == 0)
            ? ((const int[4]) {0, 1, 2, 2})
            : ((const int[1]) {2});
        const float * xdata = (rank == 0)
            ? ((const float[4]) {1.0f, 1.0f, 1.0f, 2.0f})
            : ((const float[1]) {3.0f});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_coordinate_real_single(
            &x, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        float snrm2;
        err = mtxdistmatrix_snrm2(&x, &snrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistmatrix_dnrm2(&x, &dnrm2, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistmatrix_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_scal()’ tests scaling matrices by a constant.
 */
int test_mtxdistmatrix_scal(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const float * xdata = (rank == 0)
            ? ((const float[6]) {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 2.0f})
            : ((const float[3]) {0.0f, 0.0f, 3.0f});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_real_single(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistmatrix_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[2]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[3]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[4]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistmatrix_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[2]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[3]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[4]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.array.data.real_single[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.array.data.real_single[2]);
        }
        mtxdistmatrix_free(&x);
    }

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * xdata = (rank == 0)
            ? ((const double[6]) {1.0, 1.0, 1.0, 0.0, 0.0, 2.0})
            : ((const double[3]) {0.0, 0.0, 3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_real_double(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistmatrix_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[2]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.real_double[3]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.real_double[4]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(6.0, x.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistmatrix_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_double[2]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_double[3]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_double[4]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.array.data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.array.data.real_double[2]);
        }
        mtxdistmatrix_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        const int * rowidx = (rank == 0)
            ? ((const int[4]) {0, 0, 0, 1})
            : ((const int[1]) {0});
        const int * colidx = (rank == 0)
            ? ((const int[4]) {0, 1, 2, 2})
            : ((const int[1]) {2});
        const float * xdata = (rank == 0)
            ? ((const float[4]) {1.0f, 1.0f, 1.0f, 2.0f})
            : ((const float[1]) {3.0f});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_coordinate_real_single(
            &x, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistmatrix_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[2]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.real_single[0]);
        }
        err = mtxdistmatrix_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[2]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.real_single[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.real_single[0]);
        }
        mtxdistmatrix_free(&x);
    }

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        const int * rowidx = (rank == 0)
            ? ((const int[4]) {0, 0, 0, 1})
            : ((const int[1]) {0});
        const int * colidx = (rank == 0)
            ? ((const int[4]) {0, 1, 2, 2})
            : ((const int[1]) {2});
        const double * xdata = (rank == 0)
            ? ((const double[4]) {1.0, 1.0, 1.0, 2.0})
            : ((const double[1]) {3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_coordinate_real_double(
            &x, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistmatrix_sscal(2.0f, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.coordinate.data.real_double[2]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.coordinate.data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, x.interior.storage.coordinate.data.real_double[0]);
        }
        err = mtxdistmatrix_dscal(2.0, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.coordinate.data.real_double[2]);
            TEST_ASSERT_EQ(8.0, x.interior.storage.coordinate.data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(12.0, x.interior.storage.coordinate.data.real_double[0]);
        }
        mtxdistmatrix_free(&x);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistmatrix_axpy()’ tests multiplying a matrix by a constant
 * and adding the result to another matrix.
 */
int test_mtxdistmatrix_axpy(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        const double * xdata = (rank == 0)
            ? ((const double[6]) {1.0, 1.0, 1.0, 0.0, 0.0, 2.0})
            : ((const double[3]) {0.0, 0.0, 3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_array_real_double(
            &x, num_local_rows, num_local_columns,
            xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        const double * ydata = (rank == 0)
            ? ((const double[6]) {2.0, 1.0, 0.0, 0.0, 0.0, 2.0})
            : ((const double[3]) {0.0, 0.0, 1.0});
        struct mtxdistmatrix y;
        err = mtxdistmatrix_init_array_real_double(
            &y, num_local_rows, num_local_columns,
            ydata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        err = mtxdistmatrix_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ( 3.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ( 2.0, y.interior.storage.array.data.real_double[2]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[3]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[4]);
            TEST_ASSERT_EQ( 6.0, y.interior.storage.array.data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ( 7.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistmatrix_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ( 5.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ( 4.0, y.interior.storage.array.data.real_double[2]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[3]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[4]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.array.data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistmatrix_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ( 9.0, y.interior.storage.array.data.real_double[2]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[3]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[4]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.array.data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistmatrix_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(19.0, y.interior.storage.array.data.real_double[2]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[3]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[4]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.array.data.real_double[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ( 0.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.array.data.real_double[2]);
        }
        mtxdistmatrix_free(&y);
        mtxdistmatrix_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int num_local_rows = rank == 0 ? 2 : 1;
        int num_local_columns = 3;
        int num_local_nonzeros = rank == 0 ? 4 : 1;
        const int * rowidx = (rank == 0)
            ? ((const int[4]) {0, 0, 0, 1})
            : ((const int[1]) {0});
        const int * colidx = (rank == 0)
            ? ((const int[4]) {0, 1, 2, 2})
            : ((const int[1]) {2});
        const double * xdata = (rank == 0)
            ? ((const double[4]) {1.0, 1.0, 1.0, 2.0})
            : ((const double[1]) {3.0});
        struct mtxdistmatrix x;
        err = mtxdistmatrix_init_coordinate_real_double(
            &x, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, xdata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        const double * ydata = (rank == 0)
            ? ((const double[4]) {2.0, 1.0, 0.0, 2.0})
            : ((const double[1]) {1.0});
        struct mtxdistmatrix y;
        err = mtxdistmatrix_init_coordinate_real_double(
            &y, num_local_rows, num_local_columns, num_local_nonzeros,
            rowidx, colidx, ydata, NULL, NULL, comm, 0, 0, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistmatrix_saxpy(2.0f, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 4.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ( 3.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ( 2.0, y.interior.storage.coordinate.data.real_double[2]);
            TEST_ASSERT_EQ( 6.0, y.interior.storage.coordinate.data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 7.0, y.interior.storage.coordinate.data.real_double[0]);
        }
        err = mtxdistmatrix_daxpy(2.0, &x, &y, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ( 6.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ( 5.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ( 4.0, y.interior.storage.coordinate.data.real_double[2]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.coordinate.data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.real_double[0]);
        }
        err = mtxdistmatrix_saypx(2.0f, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ( 9.0, y.interior.storage.coordinate.data.real_double[2]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.coordinate.data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(29.0, y.interior.storage.coordinate.data.real_double[0]);
        }
        err = mtxdistmatrix_daypx(2.0, &y, &x, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(19.0, y.interior.storage.coordinate.data.real_double[2]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.coordinate.data.real_double[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(61.0, y.interior.storage.coordinate.data.real_double[0]);
        }
        mtxdistmatrix_free(&y);
        mtxdistmatrix_free(&x);
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
    TEST_SUITE_BEGIN("Running tests for distributed matrices\n");
    TEST_RUN(test_mtxdistmatrix_from_mtxfile);
    TEST_RUN(test_mtxdistmatrix_to_mtxfile);
    TEST_RUN(test_mtxdistmatrix_from_mtxdistfile);
    TEST_RUN(test_mtxdistmatrix_to_mtxdistfile);
    TEST_RUN(test_mtxdistmatrix_alloc_row_vector);
    TEST_RUN(test_mtxdistmatrix_dot);
    TEST_RUN(test_mtxdistmatrix_nrm2);
    TEST_RUN(test_mtxdistmatrix_scal);
    TEST_RUN(test_mtxdistmatrix_axpy);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
