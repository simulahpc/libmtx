/* This file is part of libmtx.
 *
 * Copyright (C) 2022 James D. Trotter
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
 * Last modified: 2022-01-03
 *
 * Unit tests for distributed Matrix Market files.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/mtxdistfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/matrix/distmatrix.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistmatrix";

/**
 * `test_mtxdistmatrix_gemv()' tests multiplying a matrix by a vector.
 */
int test_mtxdistmatrix_gemv(void)
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
    err = mtxmpierror_alloc(&mpierror, comm, NULL);
    if (err)
        MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        struct mtxdistmatrix A;
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
        mtxdistmatrix_free(&A);
    }

#if 0
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
#endif
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
    TEST_SUITE_BEGIN("Running tests for distributed matrices\n");
    TEST_RUN(test_mtxdistmatrix_gemv);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
