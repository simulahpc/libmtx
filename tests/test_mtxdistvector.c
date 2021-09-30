/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-09-22
 *
 * Unit tests for distributed Matrix Market files.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/vector/distvector.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistvector";

/**
 * `test_mtxdistvector_from_mtxfile()' tests converting Matrix Market
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

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxfile(
            &mtxdistvector, &srcmtxfile, comm, root, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
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
        int num_rows = 6;
        const struct mtxfile_vector_coordinate_real_double mtxdata[] =
            {{1,1.0}, {2,2.0}, {3,3.0}, {4,4.0}, {6,6.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_vector_coordinate_real_double(
            &srcmtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxfile(
            &mtxdistvector, &srcmtxfile, comm, root, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxvector_coordinate, mtxdistvector.interior.type);
        const struct mtxvector_coordinate * interior =
            &mtxdistvector.interior.storage.coordinate;
        TEST_ASSERT_EQ(mtx_field_real, interior->field);
        TEST_ASSERT_EQ(mtx_double, interior->precision);
        TEST_ASSERT_EQ(6, interior->size);
        TEST_ASSERT_EQ(rank == 0 ? 3 : 2, interior->num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, interior->indices[0]);
            TEST_ASSERT_EQ(2, interior->indices[1]);
            TEST_ASSERT_EQ(3, interior->indices[2]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4, interior->indices[0]);
            TEST_ASSERT_EQ(6, interior->indices[1]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[1]);
        }
        mtxdistvector_free(&mtxdistvector);
        mtxfile_free(&srcmtxfile);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
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
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
