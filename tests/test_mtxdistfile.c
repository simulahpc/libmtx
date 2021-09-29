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
#include <libmtx/util/partition.h>
#include <libmtx/mtxfile/mtxdistfile.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistfile";

/**
 * `test_mtxdistfile_from_mtxfile()' tests converting Matrix Market
 * files stored on a single process to distributed Matrix Market
 * files.
 */
int test_mtxdistfile_from_mtxfile(void)
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
        int num_rows = 3;
        int num_columns = 3;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_matrix_array_real_double(
            &srcmtxfile, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, comm, root, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(rank == 0 ? 2 : 1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[0], 1.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[1], 2.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[2], 3.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[3], 4.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[4], 5.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[5], 6.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[0], 7.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[1], 8.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[2], 9.0);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxfile_free(&srcmtxfile);
    }

    {
        int num_rows = 8;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_vector_array_real_double(
            &srcmtxfile, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, comm, root, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(8, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(4, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[0], 1.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[1], 2.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[2], 3.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[3], 4.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[0], 5.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[1], 6.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[2], 7.0);
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[3], 8.0);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxfile_free(&srcmtxfile);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {4, 4, 4.0}, {3, 3, 3.0}, {2, 2, 2.0}, {1, 1, 1.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, comm, root, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(4, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(4, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(4, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(4, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(4, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxfile->precision);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxfile->data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(  4, data[0].i); TEST_ASSERT_EQ(   4, data[0].j);
            TEST_ASSERT_EQ(4.0, data[0].a);
            TEST_ASSERT_EQ(  3, data[1].i); TEST_ASSERT_EQ(   3, data[1].j);
            TEST_ASSERT_EQ(3.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(  2, data[0].i); TEST_ASSERT_EQ(   2, data[0].j);
            TEST_ASSERT_EQ(2.0, data[0].a);
            TEST_ASSERT_EQ(  1, data[1].i); TEST_ASSERT_EQ(   1, data[1].j);
            TEST_ASSERT_EQ(1.0, data[1].a);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxfile_free(&srcmtxfile);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistfile_fread()` tests reading Matrix Market files from a
 * stream and distributing them among MPI processes.
 */
int test_mtxdistfile_fread(void)
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
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL,
            comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_single[0], 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_single[0], 1.6f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_double;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[0], 1.5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_real_double[0], 1.6);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_complex_single[0][0], 1.5f);
            TEST_ASSERT_EQ(mtxfile->data.array_complex_single[0][1], 2.1f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_complex_single[0][0], 1.6f);
            TEST_ASSERT_EQ(mtxfile->data.array_complex_single[0][1], 2.2f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_double;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_complex_double[0][0], 1.5);
            TEST_ASSERT_EQ(mtxfile->data.array_complex_double[0][1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_complex_double[0][0], 1.6);
            TEST_ASSERT_EQ(mtxfile->data.array_complex_double[0][1], 2.2);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_integer_single[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_integer_single[0], 3);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_double;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.array_integer_double[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.array_integer_double[0], 3);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    /*
     * Matrix coordinate formats
     */
    {
        char s[] = "%%MatrixMarket matrix coordinate real general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5\n2 3 1.5\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_real_single[0].j, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_real_single[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_real_single[0].j, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_real_single[0].a, 1.5f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix coordinate complex general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5 2.1\n"
            "2 3 -1.5 -2.1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_double;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].j, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].j, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].a[0], -1.5);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_complex_double[0].a[1], -2.1);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix coordinate integer general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 5\n2 3 6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_integer_single[0].j, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_integer_single[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_integer_single[0].j, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_integer_single[0].a, 6);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix coordinate pattern general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2\n2 3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_pattern[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_pattern[0].j, 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_pattern[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.matrix_coordinate_pattern[0].j, 3);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    /*
     * Vector coordinate formats
     */
    {
        char s[] = "%%MatrixMarket vector coordinate real general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5\n2 1.5\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_real_single[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_real_single[0].a, 1.5f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket vector coordinate complex general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5 2.1\n"
            "2 -1.5 -2.1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_double;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_complex_double[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_complex_double[0].a[0], -1.5);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_complex_double[0].a[1], -2.1);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket vector coordinate integer general\n"
            "% comment\n"
            "3 2\n"
            "3 5\n"
            "2 6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_integer_single[0].i, 2);
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_integer_single[0].a, 6);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket vector coordinate pattern general\n"
            "% comment\n"
            "3 2\n"
            "3\n"
            "2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtx_precision precision = mtx_single;
        err = mtxdistfile_fread(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        const struct mtxfile * mtxfile = &mtxdistfile.mtxfile;
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(1, mtxfile->size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile->precision);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_pattern[0].i, 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxfile->data.vector_coordinate_pattern[0].i, 2);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    mtxmpierror_free(&mpierror);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistfile_partition_rows()' tests partitioning and
 * redistributing distributed Matrix Market files.
 */
int test_mtxdistfile_partition_rows(void)
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

    int num_process_rows = 2;
    int num_process_columns = 1;
    int process_row = rank;
    int process_column = 0;

    /*
     * Array formats
     */

    {
        int num_rows = (rank == 0) ? 2 : 1;
        int num_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});
        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata,
            comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        int num_parts = 2;
        enum mtx_partition_type row_partition_type = mtx_cyclic;
        struct mtx_partition row_partition;
        err = mtx_partition_init(
            &row_partition, row_partition_type,
            src.size.num_rows, num_parts, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtx_cyclic, row_partition.type);
        TEST_ASSERT_EQ(src.size.num_rows, row_partition.size);
        TEST_ASSERT_EQ(2, row_partition.num_parts);
        TEST_ASSERT_EQ(2, row_partition.index_sets[0].size);
        TEST_ASSERT_EQ(1, row_partition.index_sets[1].size);

        int part_per_data_line[6] = {};
        int64_t data_lines_per_part_ptr[3] = {};
        int64_t data_lines_per_part[6] = {};
        err = mtxdistfile_partition_rows(
            &src, &row_partition, part_per_data_line,
            data_lines_per_part_ptr, data_lines_per_part, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(0, part_per_data_line[0]);
            TEST_ASSERT_EQ(0, part_per_data_line[1]);
            TEST_ASSERT_EQ(0, part_per_data_line[2]);
            TEST_ASSERT_EQ(1, part_per_data_line[3]);
            TEST_ASSERT_EQ(1, part_per_data_line[4]);
            TEST_ASSERT_EQ(1, part_per_data_line[5]);
            TEST_ASSERT_EQ(0, data_lines_per_part_ptr[0]);
            TEST_ASSERT_EQ(3, data_lines_per_part_ptr[1]);
            TEST_ASSERT_EQ(6, data_lines_per_part_ptr[2]);
            TEST_ASSERT_EQ(0, data_lines_per_part[0]);
            TEST_ASSERT_EQ(1, data_lines_per_part[1]);
            TEST_ASSERT_EQ(2, data_lines_per_part[2]);
            TEST_ASSERT_EQ(3, data_lines_per_part[3]);
            TEST_ASSERT_EQ(4, data_lines_per_part[4]);
            TEST_ASSERT_EQ(5, data_lines_per_part[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0, part_per_data_line[0]);
            TEST_ASSERT_EQ(0, part_per_data_line[1]);
            TEST_ASSERT_EQ(0, part_per_data_line[2]);
            TEST_ASSERT_EQ(0, data_lines_per_part_ptr[0]);
            TEST_ASSERT_EQ(3, data_lines_per_part_ptr[1]);
            TEST_ASSERT_EQ(3, data_lines_per_part_ptr[2]);
            TEST_ASSERT_EQ(0, data_lines_per_part[0]);
            TEST_ASSERT_EQ(1, data_lines_per_part[1]);
            TEST_ASSERT_EQ(2, data_lines_per_part[2]);
        }

        struct mtxdistfile dst;
        err = mtxdistfile_init_from_partition(
            &dst, &src, row_partition.num_parts,
            data_lines_per_part_ptr, data_lines_per_part, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(3, dst.size.num_rows);
        TEST_ASSERT_EQ(3, dst.size.num_columns);
        TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);

        const struct mtxfile * mtxfile = &dst.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxfile->precision);
        TEST_ASSERT_EQ(rank == 0 ? 2 : 1, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        const double * data = mtxfile->data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(7.0, data[3]);
            TEST_ASSERT_EQ(8.0, data[4]);
            TEST_ASSERT_EQ(9.0, data[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4.0, data[0]);
            TEST_ASSERT_EQ(5.0, data[1]);
            TEST_ASSERT_EQ(6.0, data[2]);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&src);
    }

    {
        int num_rows = (rank == 0) ? 2 : 6;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[6]) {3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
        struct mtxdistfile src;
        err = mtxdistfile_init_vector_array_real_double(
            &src, num_rows, srcdata, comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        int num_parts = 2;
        enum mtx_partition_type row_partition_type = mtx_block;
        struct mtx_partition row_partition;
        err = mtx_partition_init(
            &row_partition, row_partition_type,
            src.size.num_rows, num_parts, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtx_block, row_partition.type);
        TEST_ASSERT_EQ(src.size.num_rows, row_partition.size);
        TEST_ASSERT_EQ(2, row_partition.num_parts);
        TEST_ASSERT_EQ(4, row_partition.index_sets[0].size);
        TEST_ASSERT_EQ(4, row_partition.index_sets[1].size);

        int part_per_data_line[6] = {};
        int64_t data_lines_per_part_ptr[3] = {};
        int64_t data_lines_per_part[6] = {};
        err = mtxdistfile_partition_rows(
            &src, &row_partition, part_per_data_line,
            data_lines_per_part_ptr, data_lines_per_part, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(0, part_per_data_line[0]);
            TEST_ASSERT_EQ(0, part_per_data_line[1]);
            TEST_ASSERT_EQ(0, data_lines_per_part_ptr[0]);
            TEST_ASSERT_EQ(2, data_lines_per_part_ptr[1]);
            TEST_ASSERT_EQ(2, data_lines_per_part_ptr[2]);
            TEST_ASSERT_EQ(0, data_lines_per_part[0]);
            TEST_ASSERT_EQ(1, data_lines_per_part[1]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(0, part_per_data_line[0]);
            TEST_ASSERT_EQ(0, part_per_data_line[1]);
            TEST_ASSERT_EQ(1, part_per_data_line[2]);
            TEST_ASSERT_EQ(1, part_per_data_line[3]);
            TEST_ASSERT_EQ(1, part_per_data_line[4]);
            TEST_ASSERT_EQ(1, part_per_data_line[5]);
            TEST_ASSERT_EQ(0, data_lines_per_part_ptr[0]);
            TEST_ASSERT_EQ(2, data_lines_per_part_ptr[1]);
            TEST_ASSERT_EQ(6, data_lines_per_part_ptr[2]);
            TEST_ASSERT_EQ(0, data_lines_per_part[0]);
            TEST_ASSERT_EQ(1, data_lines_per_part[1]);
            TEST_ASSERT_EQ(2, data_lines_per_part[2]);
            TEST_ASSERT_EQ(3, data_lines_per_part[3]);
            TEST_ASSERT_EQ(4, data_lines_per_part[4]);
            TEST_ASSERT_EQ(5, data_lines_per_part[5]);
        }

        struct mtxdistfile dst;
        err = mtxdistfile_init_from_partition(
            &dst, &src, row_partition.num_parts,
            data_lines_per_part_ptr, data_lines_per_part, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));

        TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(8, dst.size.num_rows);
        TEST_ASSERT_EQ(-1, dst.size.num_columns);
        TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);

        const struct mtxfile * mtxfile = &dst.mtxfile;
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxfile->precision);
        TEST_ASSERT_EQ(4, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile->size.num_nonzeros);
        const double * data = mtxfile->data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(5.0, data[0]);
            TEST_ASSERT_EQ(6.0, data[1]);
            TEST_ASSERT_EQ(7.0, data[2]);
            TEST_ASSERT_EQ(8.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&src);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {4,4,4.0}, {1,1,1.0}, {3,3,3.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {4,1,4.1}, {2,2,2.0}});
        int64_t num_nonzeros = (rank == 0) ? 3 : 2;
        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            comm, &mpierror);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

        int num_parts = 2;
        enum mtx_partition_type row_partition_type = mtx_block;
        struct mtx_partition row_partition;
        err = mtx_partition_init(
            &row_partition, row_partition_type,
            src.size.num_rows, num_parts, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(mtx_block, row_partition.type);
        TEST_ASSERT_EQ(src.size.num_rows, row_partition.size);
        TEST_ASSERT_EQ(2, row_partition.num_parts);
        TEST_ASSERT_EQ(2, row_partition.index_sets[0].size);
        TEST_ASSERT_EQ(2, row_partition.index_sets[1].size);

        int part_per_data_line[3] = {};
        int64_t data_lines_per_part_ptr[3] = {};
        int64_t data_lines_per_part[3] = {};
        err = mtxdistfile_partition_rows(
            &src, &row_partition, part_per_data_line,
            data_lines_per_part_ptr, data_lines_per_part, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));
        if (rank == 0) {
            TEST_ASSERT_EQ(1, part_per_data_line[0]);
            TEST_ASSERT_EQ(0, part_per_data_line[1]);
            TEST_ASSERT_EQ(1, part_per_data_line[2]);
            TEST_ASSERT_EQ(0, data_lines_per_part_ptr[0]);
            TEST_ASSERT_EQ(1, data_lines_per_part_ptr[1]);
            TEST_ASSERT_EQ(3, data_lines_per_part_ptr[2]);
            TEST_ASSERT_EQ(1, data_lines_per_part[0]);
            TEST_ASSERT_EQ(0, data_lines_per_part[1]);
            TEST_ASSERT_EQ(2, data_lines_per_part[2]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, part_per_data_line[0]);
            TEST_ASSERT_EQ(0, part_per_data_line[1]);
            TEST_ASSERT_EQ(0, data_lines_per_part_ptr[0]);
            TEST_ASSERT_EQ(1, data_lines_per_part_ptr[1]);
            TEST_ASSERT_EQ(2, data_lines_per_part_ptr[2]);
            TEST_ASSERT_EQ(1, data_lines_per_part[0]);
            TEST_ASSERT_EQ(0, data_lines_per_part[1]);
        }

        struct mtxdistfile dst;
        err = mtxdistfile_init_from_partition(
            &dst, &src, row_partition.num_parts,
            data_lines_per_part_ptr, data_lines_per_part, &mpierror);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxmpierror_description(&mpierror)
            : mtx_strerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(4, dst.size.num_rows);
        TEST_ASSERT_EQ(4, dst.size.num_columns);
        TEST_ASSERT_EQ(5, dst.size.num_nonzeros);

        const struct mtxfile * mtxfile = &dst.mtxfile;
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile->header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile->header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile->header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile->header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxfile->precision);
        TEST_ASSERT_EQ(4, mtxfile->size.num_rows);
        TEST_ASSERT_EQ(4, mtxfile->size.num_columns);
        TEST_ASSERT_EQ(rank == 0 ? 2 : 3, mtxfile->size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxfile->data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(4, data[0].j);
            TEST_ASSERT_EQ(4.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(3, data[1].j);
            TEST_ASSERT_EQ(3.0, data[1].a);
            TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(1, data[2].j);
            TEST_ASSERT_EQ(4.1, data[2].a);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&src);
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
    TEST_SUITE_BEGIN("Running tests for distributed Matrix Market files\n");
    TEST_RUN(test_mtxdistfile_from_mtxfile);
    TEST_RUN(test_mtxdistfile_fread);
    TEST_RUN(test_mtxdistfile_partition_rows);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
