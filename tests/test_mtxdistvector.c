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
#include <libmtx/mtxfile/mtxdistfile.h>
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
            &mtxdistvector, &srcmtxfile, mtxvector_auto, comm, root, &mpierror);
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
            &mtxdistvector, &srcmtxfile, mtxvector_auto, comm, root, &mpierror);
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
            TEST_ASSERT_EQ(0, interior->indices[0]);
            TEST_ASSERT_EQ(1, interior->indices[1]);
            TEST_ASSERT_EQ(2, interior->indices[2]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, interior->indices[0]);
            TEST_ASSERT_EQ(5, interior->indices[1]);
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
 * `test_mtxdistvector_from_mtxdistfile()' tests converting
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

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = (rank == 0) ? 2 : 6;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[6]) {3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
        struct mtxdistfile src;
        err = mtxdistfile_init_vector_array_real_double(
            &src, num_rows, srcdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxdistfile(
            &mtxdistvector, &src, mtxvector_auto, comm, &mpierror);
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
        TEST_ASSERT_EQ(rank == 0 ? 2 : 6, interior->size);
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(5.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[3]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[4]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[5]);
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
        int64_t num_nonzeros = (rank == 0) ? 2 : 6;
        struct mtxdistfile src;
        err = mtxdistfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, srcdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistvector mtxdistvector;
        err = mtxdistvector_from_mtxdistfile(
            &mtxdistvector, &src, mtxvector_auto, comm, &mpierror);
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
        TEST_ASSERT_EQ(9, interior->size);
        TEST_ASSERT_EQ(rank == 0 ? 2 : 6, interior->num_nonzeros);
        if (rank == 0) {
            TEST_ASSERT_EQ(0, interior->indices[0]);
            TEST_ASSERT_EQ(1, interior->indices[1]);
            TEST_ASSERT_EQ(1.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(2.0, interior->data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, interior->indices[0]);
            TEST_ASSERT_EQ(3, interior->indices[1]);
            TEST_ASSERT_EQ(4, interior->indices[2]);
            TEST_ASSERT_EQ(5, interior->indices[3]);
            TEST_ASSERT_EQ(6, interior->indices[4]);
            TEST_ASSERT_EQ(7, interior->indices[5]);
            TEST_ASSERT_EQ(3.0, interior->data.real_double[0]);
            TEST_ASSERT_EQ(4.0, interior->data.real_double[1]);
            TEST_ASSERT_EQ(5.0, interior->data.real_double[2]);
            TEST_ASSERT_EQ(6.0, interior->data.real_double[3]);
            TEST_ASSERT_EQ(7.0, interior->data.real_double[4]);
            TEST_ASSERT_EQ(8.0, interior->data.real_double[5]);
        }
        mtxdistvector_free(&mtxdistvector);
        mtxdistfile_free(&src);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistvector_dot()' tests computing the dot products of
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

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistvector x;
        int size = (rank == 0) ? 2 : 3;
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        err = mtxdistvector_init_array_real_single(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const float * ydata = (rank == 0)
            ? ((const float[2]) {3.0f, 2.0f})
            : ((const float[3]) {1.0f, 0.0f, 1.0f});
        err = mtxdistvector_init_array_real_single(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = (rank == 0) ? 2 : 3;
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});
        err = mtxdistvector_init_array_real_double(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const double * ydata = (rank == 0)
            ? ((const double[2]) {3.0, 2.0})
            : ((const double[3]) {1.0, 0.0, 1.0});
        err = mtxdistvector_init_array_real_double(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = (rank == 0) ? 1 : 2;
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[][2]) {{1.0f, 1.0f}})
            : ((const float[][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        err = mtxdistvector_init_array_complex_single(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[][2]) {{3.0f, 2.0f}})
            : ((const float[][2]) {{1.0f, 0.0f}, {1.0f, 0.0f}});
        err = mtxdistvector_init_array_complex_single(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = (rank == 0) ? 1 : 2;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[][2]) {{1.0, 1.0}})
            : ((const double[][2]) {{1.0, 2.0}, {3.0, 0.0}});
        err = mtxdistvector_init_array_complex_double(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[][2]) {{3.0, 2.0}})
            : ((const double[][2]) {{1.0, 0.0}, {1.0, 0.0}});
        err = mtxdistvector_init_array_complex_double(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = (rank == 0) ? 2 : 3;
        const int32_t * xdata = (rank == 0)
            ? ((const int32_t[2]) {1, 1})
            : ((const int32_t[3]) {1, 2, 3});
        err = mtxdistvector_init_array_integer_single(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const int32_t * ydata = (rank == 0)
            ? ((const int32_t[2]) {3, 2})
            : ((const int32_t[3]) {1, 0, 1});
        err = mtxdistvector_init_array_integer_single(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = (rank == 0) ? 2 : 3;
        const int64_t * xdata = (rank == 0)
            ? ((const int64_t[2]) {1, 1})
            : ((const int64_t[3]) {1, 2, 3});
        err = mtxdistvector_init_array_integer_double(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const int64_t * ydata = (rank == 0)
            ? ((const int64_t[2]) {3, 2})
            : ((const int64_t[3]) {1, 0, 1});
        err = mtxdistvector_init_array_integer_double(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        int size = 12;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1, 3})
            : ((const int[3]) {5, 7, 9});
        struct mtxdistvector x;
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        err = mtxdistvector_init_coordinate_real_single(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const float * ydata = (rank == 0)
            ? ((const float[2]) {3.0f, 2.0f})
            : ((const float[3]) {1.0f, 0.0f, 1.0f});
        err = mtxdistvector_init_coordinate_real_single(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1, 3})
            : ((const int[3]) {5, 7, 9});
        struct mtxdistvector x;
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0f, 1.0f})
            : ((const double[3]) {1.0f, 2.0f, 3.0f});
        err = mtxdistvector_init_coordinate_real_double(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const double * ydata = (rank == 0)
            ? ((const double[2]) {3.0f, 2.0f})
            : ((const double[3]) {1.0f, 0.0f, 1.0f});
        err = mtxdistvector_init_coordinate_real_double(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int nnz = (rank == 0) ? 1 : 2;
        const int * idx = (rank == 0)
            ? ((const int[1]) {1})
            : ((const int[2]) {3, 5});
        struct mtxdistvector x;
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f, 1.0f}})
            : ((const float[2][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        err = mtxdistvector_init_coordinate_complex_single(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[1][2]) {{3.0f, 2.0f}})
            : ((const float[2][2]) {{1.0f, 0.0f}, {1.0f, 0.0f}});
        err = mtxdistvector_init_coordinate_complex_single(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int nnz = (rank == 0) ? 1 : 2;
        const int * idx = (rank == 0)
            ? ((const int[1]) {1})
            : ((const int[2]) {3, 5});
        struct mtxdistvector x;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0f, 1.0f}})
            : ((const double[2][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        err = mtxdistvector_init_coordinate_complex_double(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[1][2]) {{3.0f, 2.0f}})
            : ((const double[2][2]) {{1.0f, 0.0f}, {1.0f, 0.0f}});
        err = mtxdistvector_init_coordinate_complex_double(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_ERR_MPI_COLLECTIVE, err, "%s", mtx_strerror(err));
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0f, cdotu[0]); TEST_ASSERT_EQ(7.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(-3.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(5.0, zdotu[0]); TEST_ASSERT_EQ(7.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(-3.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1, 3})
            : ((const int[3]) {5, 7, 9});
        struct mtxdistvector x;
        const int32_t * xdata = (rank == 0)
            ? ((const int32_t[2]) {1.0f, 1.0f})
            : ((const int32_t[3]) {1.0f, 2.0f, 3.0f});
        err = mtxdistvector_init_coordinate_integer_single(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const int32_t * ydata = (rank == 0)
            ? ((const int32_t[2]) {3.0f, 2.0f})
            : ((const int32_t[3]) {1.0f, 0.0f, 1.0f});
        err = mtxdistvector_init_coordinate_integer_single(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        int size = 12;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1, 3})
            : ((const int[3]) {5, 7, 9});
        struct mtxdistvector x;
        const int64_t * xdata = (rank == 0)
            ? ((const int64_t[2]) {1.0f, 1.0f})
            : ((const int64_t[3]) {1.0f, 2.0f, 3.0f});
        err = mtxdistvector_init_coordinate_integer_double(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        struct mtxdistvector y;
        const int64_t * ydata = (rank == 0)
            ? ((const int64_t[2]) {3.0f, 2.0f})
            : ((const int64_t[3]) {1.0f, 0.0f, 1.0f});
        err = mtxdistvector_init_coordinate_integer_double(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float sdot;
        err = mtxdistvector_sdot(&x, &y, &sdot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, sdot);
        double ddot;
        err = mtxdistvector_ddot(&x, &y, &ddot, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, ddot);
        float cdotu[2];
        err = mtxdistvector_cdotu(&x, &y, &cdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotu[0]); TEST_ASSERT_EQ(0.0f, cdotu[1]);
        float cdotc[2];
        err = mtxdistvector_cdotc(&x, &y, &cdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0f, cdotc[0]); TEST_ASSERT_EQ(0.0f, cdotc[1]);
        double zdotu[2];
        err = mtxdistvector_zdotu(&x, &y, &zdotu, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotu[0]); TEST_ASSERT_EQ(0.0, zdotu[1]);
        double zdotc[2];
        err = mtxdistvector_zdotc(&x, &y, &zdotc, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(9.0, zdotc[0]); TEST_ASSERT_EQ(0.0, zdotc[1]);
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistvector_nrm2()' tests computing the Euclidean norm of
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

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
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
        int size = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_array_real_single(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ_MSG(4.0f, snrm2, "snrm2=%f", snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});
        int size = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_array_real_double(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const float (* data)[2] = (rank == 0)
            ? ((const float[][2]) {1.0f,1.0f})
            : ((const float[][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        int size = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_array_complex_single(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double (* data)[2] = (rank == 0)
            ? ((const double[][2]) {1.0,1.0})
            : ((const double[][2]) {{1.0,2.0}, {3.0,0.0}});
        int size = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_array_complex_double(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const int32_t * data = (rank == 0)
            ? ((const int32_t[2]) {1, 1})
            : ((const int32_t[3]) {1, 2, 3});
        int size = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_array_integer_single(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const int64_t * data = (rank == 0)
            ? ((const int64_t[2]) {1, 1})
            : ((const int64_t[3]) {1, 2, 3});
        int size = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_array_integer_double(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {5,6,9});
        const float * data = (rank == 0)
            ? ((const float[2]) {1.0f,1.0f})
            : ((const float[3]) {1.0f,2.0f,3.0f});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {5,6,9});
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0,1.0})
            : ((const double[3]) {1.0,2.0,3.0});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {5,9});
        const float (* data)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f,1.0f}})
            : ((const float[2][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {5,9});
        const double (* data)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0,1.0}})
            : ((const double[2][2]) {{1.0,2.0}, {3.0,0.0}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {5,6,9});
        const int32_t * data = (rank == 0)
            ? ((const int32_t[2]) {1,1})
            : ((const int32_t[3]) {1,2,3});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_coordinate_integer_single(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {5,6,9});
        const int64_t * data = (rank == 0)
            ? ((const int64_t[2]) {1,1})
            : ((const int64_t[3]) {1,2,3});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_coordinate_integer_double(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 24;
        const int * indices = (rank == 0)
            ? ((const int[7]) {0,2,3,4,5,6,7})
            : ((const int[9]) {9,10,11,13,14,17,20,21,23});
        size_t num_nonzeros = (rank == 0) ? 7 : 9;
        err = mtxdistvector_init_coordinate_pattern(
            &x, size, num_nonzeros, indices, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        float snrm2;
        err = mtxdistvector_snrm2(&x, &snrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0f, snrm2);
        double dnrm2;
        err = mtxdistvector_dnrm2(&x, &dnrm2, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(4.0, dnrm2);
        mtxdistvector_free(&x);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistvector_scal()' tests scaling vectors by a constant.
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

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
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
        int size = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_array_real_single(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        if (rank == 0) {
            TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.array.data.real_single[2]);
        }
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});
        int size = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_array_real_double(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(6.0, x.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(8.0, x.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(12.0, x.interior.storage.array.data.real_double[2]);
        }
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const float (* data)[2] = (rank == 0)
            ? ((const float[][2]) {1.0f,1.0f})
            : ((const float[][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        int size = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_array_complex_single(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.array.data.complex_single[1][1]);
        }
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        const double (* data)[2] = (rank == 0)
            ? ((const double[][2]) {1.0,1.0})
            : ((const double[][2]) {{1.0,2.0}, {3.0,0.0}});
        int size = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_array_complex_double(&x, size, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(6.0, x.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, x.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(8.0, x.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(12.0, x.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, x.interior.storage.array.data.complex_double[1][1]);
        }
        mtxdistvector_free(&x);
    }

    /*
     * Coordinate formats
     */

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {5,6,9});
        const float * data = (rank == 0)
            ? ((const float[2]) {1.0f,1.0f})
            : ((const float[3]) {1.0f,2.0f,3.0f});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_coordinate_real_single(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.real_single[2]);
        }
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[2]) {0,3})
            : ((const int[3]) {5,6,9});
        const double * data = (rank == 0)
            ? ((const double[2]) {1.0,1.0})
            : ((const double[3]) {1.0,2.0,3.0});
        size_t num_nonzeros = (rank == 0) ? 2 : 3;
        err = mtxdistvector_init_coordinate_real_double(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.real_double[2]);
        }
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {5,9});
        const float (* data)[2] = (rank == 0)
            ? ((const float[1][2]) {{1.0f,1.0f}})
            : ((const float[2][2]) {{1.0f,2.0f}, {3.0f,0.0f}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_coordinate_complex_single(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_single[1][1]);
        }
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        int size = 12;
        const int * indices = (rank == 0)
            ? ((const int[1]) {0})
            : ((const int[2]) {5,9});
        const double (* data)[2] = (rank == 0)
            ? ((const double[1][2]) {{1.0,1.0}})
            : ((const double[2][2]) {{1.0,2.0}, {3.0,0.0}});
        size_t num_nonzeros = (rank == 0) ? 1 : 2;
        err = mtxdistvector_init_coordinate_complex_double(
            &x, size, num_nonzeros, indices, data, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_sscal(2.0f, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(6.0f, x.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_dscal(2.0, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, x.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(8.0f, x.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(12.0f, x.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0f, x.interior.storage.coordinate.data.complex_double[1][1]);
        }
        mtxdistvector_free(&x);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistvector_axpy()' tests multiplying a vector by a constant
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

    struct mtxmpierror mpierror;
    err = mtxmpierror_alloc(&mpierror, comm);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = (rank == 0) ? 2 : 3;
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        const float * ydata = (rank == 0)
            ? ((const float[2]) {2.0f, 1.0f})
            : ((const float[3]) {0.0f, 2.0f, 1.0f});
        err = mtxdistvector_init_array_real_single(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_array_real_single(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.array.data.real_single[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.array.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.array.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.array.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.array.data.real_single[2]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = (rank == 0) ? 2 : 3;
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0, 1.0})
            : ((const double[3]) {1.0, 2.0, 3.0});
        const double * ydata = (rank == 0)
            ? ((const double[2]) {2.0, 1.0})
            : ((const double[3]) {0.0, 2.0, 1.0});
        err = mtxdistvector_init_array_real_double(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_array_real_double(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.array.data.real_double[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.array.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.array.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.array.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.array.data.real_double[2]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = (rank == 0) ? 1 : 2;
        const float (* xdata)[2] = (rank == 0)
            ? ((const float[][2]) {{1.0f, 1.0f}})
            : ((const float[][2]) {{1.0f, 2.0f}, {3.0f, 0.0f}});
        const float (* ydata)[2] = (rank == 0)
            ? ((const float[][2]) {{2.0f, 1.0f}})
            : ((const float[][2]) {{0.0f, 2.0f}, {1.0f, 0.0f}});
        err = mtxdistvector_init_array_complex_single(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_array_complex_single(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.array.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.array.data.complex_single[0][0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.array.data.complex_single[0][1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.array.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.array.data.complex_single[1][1]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = (rank == 0) ? 1 : 2;
        const double (* xdata)[2] = (rank == 0)
            ? ((const double[][2]) {{1.0, 1.0}})
            : ((const double[][2]) {{1.0, 2.0}, {3.0, 0.0}});
        const double (* ydata)[2] = (rank == 0)
            ? ((const double[][2]) {{2.0, 1.0}})
            : ((const double[][2]) {{0.0, 2.0}, {1.0, 0.0}});
        err = mtxdistvector_init_array_complex_double(&x, size, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_array_complex_double(&y, size, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.array.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.array.data.complex_double[0][0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.array.data.complex_double[0][1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.array.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.array.data.complex_double[1][1]);
        }
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
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1, 3})
            : ((const int[3]) {5, 7, 9});
        const float * xdata = (rank == 0)
            ? ((const float[2]) {1.0f, 1.0f})
            : ((const float[3]) {1.0f, 2.0f, 3.0f});
        const float * ydata = (rank == 0)
            ? ((const float[2]) {2.0f, 1.0f})
            : ((const float[3]) {0.0f, 2.0f, 1.0f});
        err = mtxdistvector_init_coordinate_real_single(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_coordinate_real_single(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.coordinate.data.real_single[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.coordinate.data.real_single[0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.coordinate.data.real_single[1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.coordinate.data.real_single[2]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
        int nnz = (rank == 0) ? 2 : 3;
        const int * idx = (rank == 0)
            ? ((const int[2]) {1, 3})
            : ((const int[3]) {5, 7, 9});
        const double * xdata = (rank == 0)
            ? ((const double[2]) {1.0f, 1.0f})
            : ((const double[3]) {1.0f, 2.0f, 3.0f});
        const double * ydata = (rank == 0)
            ? ((const double[2]) {2.0f, 1.0f})
            : ((const double[3]) {0.0f, 2.0f, 1.0f});
        err = mtxdistvector_init_coordinate_real_double(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_coordinate_real_double(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.coordinate.data.real_double[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.coordinate.data.real_double[0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.coordinate.data.real_double[1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.coordinate.data.real_double[2]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
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
        err = mtxdistvector_init_coordinate_complex_single(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_coordinate_complex_single(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(3.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(7.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(5.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(10.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(11.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(22.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(29.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(23.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0f, y.interior.storage.coordinate.data.complex_single[0][0]);
            TEST_ASSERT_EQ(46.0f, y.interior.storage.coordinate.data.complex_single[0][1]);
            TEST_ASSERT_EQ(61.0f, y.interior.storage.coordinate.data.complex_single[1][0]);
            TEST_ASSERT_EQ(0.0f, y.interior.storage.coordinate.data.complex_single[1][1]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
    }

    {
        struct mtxdistvector x;
        struct mtxdistvector y;
        int size = 12;
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
        err = mtxdistvector_init_coordinate_complex_double(
            &x, size, nnz, idx, xdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_init_coordinate_complex_double(
            &y, size, nnz, idx, ydata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        err = mtxdistvector_saxpy(2.0f, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(3.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(7.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_daxpy(2.0, &x, &y, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(6.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(5.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(10.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_saypx(2.0f, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(13.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(11.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(9.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(22.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(29.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        err = mtxdistvector_daypx(2.0, &y, &x, NULL, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(27.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(23.0, y.interior.storage.coordinate.data.complex_double[0][1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(19.0, y.interior.storage.coordinate.data.complex_double[0][0]);
            TEST_ASSERT_EQ(46.0, y.interior.storage.coordinate.data.complex_double[0][1]);
            TEST_ASSERT_EQ(61.0, y.interior.storage.coordinate.data.complex_double[1][0]);
            TEST_ASSERT_EQ(0.0, y.interior.storage.coordinate.data.complex_double[1][1]);
        }
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
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
    TEST_RUN(test_mtxdistvector_from_mtxdistfile);
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
