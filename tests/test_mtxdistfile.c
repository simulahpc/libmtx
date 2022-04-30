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
#include <libmtx/util/partition.h>

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

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(9, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(5, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, mtxdistfile.partition.part_sizes[1]);
        union mtxfiledata * data = &mtxdistfile.data;
        if (rank == 0) {
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
            TEST_ASSERT_EQ(data->array_real_double[4], 5.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(data->array_real_double[0], 6.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 8.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 9.0);
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
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(8, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(8, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(4, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, mtxdistfile.partition.part_sizes[1]);
        union mtxfiledata * data = &mtxdistfile.data;
        if (rank == 0) {
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(data->array_real_double[0], 5.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 6.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 8.0);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxfile_free(&srcmtxfile);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 5;
        int num_columns = 5;
        const struct mtxfile_matrix_coordinate_real_double mtxdata[] = {
            {4, 4, 4.0}, {3, 3, 3.0}, {2, 2, 2.0}, {1, 1, 1.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_matrix_coordinate_real_double(
            &srcmtxfile, mtxfile_general,
            num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(5, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(5, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(4, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(4, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtxdistfile.data.matrix_coordinate_real_double;
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

    /*
     * Vector coordinate formats
     */

    {
        int num_rows = 6;
        const struct mtxfile_vector_coordinate_real_double mtxdata[] = {
            {5, 5.0}, {3, 3.0}, {2, 2.0}, {1, 1.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile srcmtxfile;
        err = mtxfile_init_vector_coordinate_real_double(
            &srcmtxfile, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_from_mtxfile(
            &mtxdistfile, &srcmtxfile, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(6, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(4, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, mtxdistfile.precision);
        TEST_ASSERT_EQ(4, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.part_sizes[1]);
        const struct mtxfile_vector_coordinate_real_double * data =
            mtxdistfile.data.vector_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(5, data[0].i); TEST_ASSERT_EQ(5.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(3.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(2.0, data[0].a);
            TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(1.0, data[1].a);
        }
        mtxdistfile_free(&mtxdistfile);
        mtxfile_free(&srcmtxfile);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistfile_fread_shared()` tests reading Matrix Market files from a
 * stream and distributing them among MPI processes.
 */
int test_mtxdistfile_fread_shared(void)
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
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL,
            comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_real_single[0], 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_real_single[0], 1.6f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_real_double[0], 1.5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_real_double[0], 1.6);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_single[0][0], 1.5f);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_single[0][1], 2.1f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_single[0][0], 1.6f);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_single[0][1], 2.2f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_double[0][0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_double[0][1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_double[0][0], 1.6);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_double[0][1], 2.2);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_integer_single[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_integer_single[0], 3);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_integer_double[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.array_integer_double[0], 3);
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
            "3 2 1.5\n2 3 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].j, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].a, 1.5f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix coordinate complex general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5 2.1\n"
            "2 3 -1.5 -2.1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].j, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].a[0], -1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].a[1], -2.1);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix coordinate integer general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 5\n2 3 6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].j, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].a, 6);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket matrix coordinate pattern general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2\n2 3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_pattern[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_pattern[0].j, 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_pattern[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_pattern[0].j, 3);
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
            "3 1.5\n2 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_real_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_real_single[0].a, 1.5f);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket vector coordinate complex general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5 2.1\n"
            "2 -1.5 -2.1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].a[0], -1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].a[1], -2.1);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket vector coordinate integer general\n"
            "% comment\n"
            "3 2\n"
            "3 5\n"
            "2 6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_integer_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_integer_single[0].a, 6);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    {
        char s[] = "%%MatrixMarket vector coordinate pattern general\n"
            "% comment\n"
            "3 2\n"
            "3\n"
            "2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxdistfile mtxdistfile;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile_fread_shared(
            &mtxdistfile, precision,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        TEST_ASSERT_EQ_MSG(
            strlen(s), bytes_read,
            "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.partition.size);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[0]);
        TEST_ASSERT_EQ(1, mtxdistfile.partition.part_sizes[1]);
        if (rank == 0) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_pattern[0].i, 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_pattern[0].i, 2);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistfile_fwrite_shared()' tests writing distributed Matrix
 * Market files to a stream.
 */
int test_mtxdistfile_fwrite_shared(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 3 : 3;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_init_matrix_array_real_double(
            &mtxdistfile, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxdistfile_fwrite_shared(
            &mtxdistfile, f, "%.1f", &bytes_written, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        fflush(f);
        if (rank == root) {
            char expected[] =
                "%%MatrixMarket matrix array real general\n"
                "3 3\n"
                "1.0\n" "2.0\n" "3.0\n"
                "4.0\n" "5.0\n" "6.0\n"
                "7.0\n" "8.0\n" "9.0\n";
            TEST_ASSERT_STREQ_MSG(
                expected, buf, "expected:\n%s\nactual:\n%s",
                expected, buf);
        }
        fclose(f);
        mtxdistfile_free(&mtxdistfile);
    }

    {
        int num_rows = 5;
        int64_t size = 5;
        const double * srcdata = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[3]) {3.0, 4.0, 5.0});

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_init_vector_array_real_double(
            &mtxdistfile, num_rows, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxdistfile_fwrite_shared(
            &mtxdistfile, f, "%.1f", &bytes_written, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        fflush(f);
        if (rank == root) {
            char expected[] =
                "%%MatrixMarket vector array real general\n"
                "5\n"
                "1.0\n" "2.0\n" "3.0\n" "4.0\n" "5.0\n";
            TEST_ASSERT_STREQ_MSG(
                expected, buf, "expected:\n%s\nactual:\n%s",
                expected, buf);
        }
        fclose(f);
        mtxdistfile_free(&mtxdistfile);
        mtxpartition_free(&partition);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        int64_t num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {1,1,1.0}, {2,2,2.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {3,3,3.0}, {4,1,4.1}, {4,4,4.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &mtxdistfile, mtxfile_general, num_rows, num_columns, num_nonzeros,
            srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxdistfile_fwrite_shared(
            &mtxdistfile, f, "%.1f", &bytes_written, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        fflush(f);
        if (rank == root) {
            char expected[] =
                "%%MatrixMarket matrix coordinate real general\n"
                "4 4 5\n"
                "1 1 1.0\n"
                "2 2 2.0\n"
                "3 3 3.0\n"
                "4 1 4.1\n" "4 4 4.0\n";
            TEST_ASSERT_STREQ_MSG(
                expected, buf, "expected:\n%s\nactual:\n%s",
                expected, buf);
        }
        fclose(f);
        mtxdistfile_free(&mtxdistfile);
        mtxpartition_free(&partition);
    }

    /*
     * Vector coordinate formats
     */

    {
        int num_rows = 6;
        int64_t num_nonzeros = 5;
        const struct mtxfile_vector_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_double[2]) {
                    {1,1.0}, {2,2.0}})
            : ((const struct mtxfile_vector_coordinate_real_double[3]) {
                    {3,3.0}, {1,1.0}, {6,6.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {2,3};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile mtxdistfile;
        err = mtxdistfile_init_vector_coordinate_real_double(
            &mtxdistfile, num_rows, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxdistfile_fwrite_shared(
            &mtxdistfile, f, "%.1f", &bytes_written, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        fflush(f);
        if (rank == root) {
            char expected[] =
                "%%MatrixMarket vector coordinate real general\n"
                "6 5\n"
                "1 1.0\n" "2 2.0\n"
                "3 3.0\n" "1 1.0\n" "6 6.0\n";
            TEST_ASSERT_STREQ_MSG(
                expected, buf, "expected:\n%s\nactual:\n%s",
                expected, buf);
        }
        fclose(f);
        mtxdistfile_free(&mtxdistfile);
        mtxpartition_free(&partition);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistfile_partition()’ tests partitioning and
 * redistributing the entries of a distributed Matrix Market file.
 */
int test_mtxdistfile_partition(void)
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
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 3 : 3;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[2];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(6, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(5, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(2.0, data[1]);
                TEST_ASSERT_EQ(3.0, data[2]);
                TEST_ASSERT_EQ(4.0, data[3]);
                TEST_ASSERT_EQ(5.0, data[4]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(6.0, data[0]);
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(1, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(3, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 1) {
                TEST_ASSERT_EQ(7.0, data[0]);
                TEST_ASSERT_EQ(8.0, data[1]);
                TEST_ASSERT_EQ(9.0, data[2]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 3 : 3;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_cyclic;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[2];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(6, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(5, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(2.0, data[1]);
                TEST_ASSERT_EQ(3.0, data[2]);
                TEST_ASSERT_EQ(7.0, data[3]);
                TEST_ASSERT_EQ(8.0, data[4]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(9.0, data[0]);
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(1, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(3, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(4.0, data[0]);
                TEST_ASSERT_EQ(5.0, data[1]);
                TEST_ASSERT_EQ(6.0, data[2]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 3 : 3;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        int num_col_parts = 2;
        enum mtxpartitioning colpart_type = mtx_block;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[2];
        err = mtxdistfile_partition(dsts, &src, NULL, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(3, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(6, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(5, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(2.0, data[1]);
                TEST_ASSERT_EQ(4.0, data[2]);
                TEST_ASSERT_EQ(5.0, data[3]);
                TEST_ASSERT_EQ(7.0, data[4]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(8.0, data[0]);
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(3, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(3, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(3.0, data[0]);
                TEST_ASSERT_EQ(6.0, data[1]);
                TEST_ASSERT_EQ(9.0, data[2]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        mtxpartition_free(&colpart);
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 3 : 3;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_col_parts = 2;
        enum mtxpartitioning colpart_type = mtx_block;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts*num_col_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(4, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(4, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(0, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(2.0, data[1]);
                TEST_ASSERT_EQ(4.0, data[2]);
                TEST_ASSERT_EQ(5.0, data[3]);
            } else if (rank == 1) {
                /* empty */
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(2, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(2, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(3.0, data[0]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(6.0, data[0]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[2].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[2].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[2].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[2].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[2].precision);
            TEST_ASSERT_EQ(1, dsts[2].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[2].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_nonzeros);
            TEST_ASSERT_EQ(2, dsts[2].partition.size);
            TEST_ASSERT_EQ(2, dsts[2].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[2].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[2].partition.part_sizes[1]);
            const double * data = dsts[2].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(7.0, data[0]);
                TEST_ASSERT_EQ(8.0, data[1]);
            }
            mtxdistfile_free(&dsts[2]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[3].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[3].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[3].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[3].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[3].precision);
            TEST_ASSERT_EQ(1, dsts[3].size.num_rows);
            TEST_ASSERT_EQ(1, dsts[3].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[3].size.num_nonzeros);
            TEST_ASSERT_EQ(1, dsts[3].partition.size);
            TEST_ASSERT_EQ(2, dsts[3].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[3].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[3].partition.part_sizes[1]);
            const double * data = dsts[3].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(9.0, data[0]);
            }
            mtxdistfile_free(&dsts[3]);
        }
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 3 : 3;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[3]) {0, 1, 2})
            : ((const int64_t[3]) {0, 1, 2});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_col_parts = 2;
        enum mtxpartitioning colpart_type = mtx_cyclic;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts*num_col_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(4, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(4, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(0, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(3.0, data[1]);
                TEST_ASSERT_EQ(4.0, data[2]);
                TEST_ASSERT_EQ(6.0, data[3]);
            } else if (rank == 1) {
                /* empty */
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(2, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(2, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(2.0, data[0]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(5.0, data[0]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[2].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[2].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[2].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[2].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[2].precision);
            TEST_ASSERT_EQ(1, dsts[2].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[2].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_nonzeros);
            TEST_ASSERT_EQ(2, dsts[2].partition.size);
            TEST_ASSERT_EQ(2, dsts[2].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[2].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[2].partition.part_sizes[1]);
            const double * data = dsts[2].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(7.0, data[0]);
                TEST_ASSERT_EQ(9.0, data[1]);
            }
            mtxdistfile_free(&dsts[2]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[3].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[3].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[3].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[3].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[3].precision);
            TEST_ASSERT_EQ(1, dsts[3].size.num_rows);
            TEST_ASSERT_EQ(1, dsts[3].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[3].size.num_nonzeros);
            TEST_ASSERT_EQ(1, dsts[3].partition.size);
            TEST_ASSERT_EQ(2, dsts[3].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[3].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[3].partition.part_sizes[1]);
            const double * data = dsts[3].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(8.0, data[0]);
            }
            mtxdistfile_free(&dsts[3]);
        }
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 8;
        int64_t size = 8;
        const double * srcdata = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 3.0, 4.0})
            : ((const double[4]) {5.0, 6.0, 7.0, 8.0});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {4,4};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_array_real_double(
            &src, num_rows, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 3;
        enum mtxpartitioning rowparttype = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowparttype, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(3, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(3, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(0, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(2.0, data[1]);
                TEST_ASSERT_EQ(3.0, data[2]);
            } else if (rank == 1) {
                /* empty */
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(3, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(4.0, data[0]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(5.0, data[0]);
                TEST_ASSERT_EQ(6.0, data[1]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[2].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[2].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[2].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[2].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[2].precision);
            TEST_ASSERT_EQ(2, dsts[2].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_nonzeros);
            TEST_ASSERT_EQ(2, dsts[2].partition.size);
            TEST_ASSERT_EQ(2, dsts[2].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[2].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[2].partition.part_sizes[1]);
            const double * data = dsts[2].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(7.0, data[0]);
                TEST_ASSERT_EQ(8.0, data[1]);
            }
            mtxdistfile_free(&dsts[2]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    {
        int num_rows = 8;
        int64_t size = 8;
        const double * srcdata = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 3.0, 4.0})
            : ((const double[4]) {5.0, 6.0, 7.0, 8.0});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {4,4};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, size, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_array_real_double(
            &src, num_rows, srcdata, &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 3;
        enum mtxpartitioning rowparttype = mtx_cyclic;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowparttype, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(3, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(3, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(0, dsts[0].partition.part_sizes[1]);
            const double * data = dsts[0].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(1.0, data[0]);
                TEST_ASSERT_EQ(4.0, data[1]);
                TEST_ASSERT_EQ(7.0, data[2]);
            } else if (rank == 1) {
                /* empty */
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(3, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[1].partition.part_sizes[1]);
            const double * data = dsts[1].data.array_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(2.0, data[0]);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(5.0, data[0]);
                TEST_ASSERT_EQ(8.0, data[1]);
            }
            mtxdistfile_free(&dsts[1]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[2].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[2].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[2].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[2].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[2].precision);
            TEST_ASSERT_EQ(2, dsts[2].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_nonzeros);
            TEST_ASSERT_EQ(2, dsts[2].partition.size);
            TEST_ASSERT_EQ(2, dsts[2].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[2].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[2].partition.part_sizes[1]);
            const double * data = dsts[2].data.array_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(3.0, data[0]);
                TEST_ASSERT_EQ(6.0, data[1]);
            }
            mtxdistfile_free(&dsts[2]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        int64_t num_nonzeros = 9;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[4]) {
                    {4,4,16.0}, {3,3, 9.0}, {2,2, 4.0}, {1,1, 1.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[5]) {
                    {1,2, 2.0}, {2,1, 5.0}, {2,3, 7.0}, {3,4,10.0}, {4,2,14.0}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {4,5};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(4, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(5, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(5, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(4, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(1, dsts[0].partition.part_sizes[1]);
            const struct mtxfile_matrix_coordinate_real_double * data =
                dsts[0].data.matrix_coordinate_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(2, data[0].j);
                TEST_ASSERT_EQ(4.0, data[0].a);
                TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(1, data[1].j);
                TEST_ASSERT_EQ(1.0, data[1].a);
                TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(2, data[2].j);
                TEST_ASSERT_EQ(2.0, data[2].a);
                TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(1, data[3].j);
                TEST_ASSERT_EQ(5.0, data[3].a);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
                TEST_ASSERT_EQ(7.0, data[0].a);
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(2, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(4, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(4, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(4, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(0, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(4, dsts[1].partition.part_sizes[1]);
            const struct mtxfile_matrix_coordinate_real_double * data =
                dsts[1].data.matrix_coordinate_real_double;
            if (rank == 0) {
                /* empty */
            } else if (rank == 1) {
                TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(4, data[0].j);
                TEST_ASSERT_EQ(16.0, data[0].a);
                TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(3, data[1].j);
                TEST_ASSERT_EQ(9.0, data[1].a);
                TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(4, data[2].j);
                TEST_ASSERT_EQ(10.0, data[2].a);
                TEST_ASSERT_EQ(2, data[3].i); TEST_ASSERT_EQ(2, data[3].j);
                TEST_ASSERT_EQ(14.0, data[3].a);
            }
            mtxdistfile_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    /*
     * Vector coordinate formats
     */

    {
        int num_rows = 9;
        int64_t num_nonzeros = 6;
        const struct mtxfile_vector_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_double[4]) {
                    {4,4.0}, {3,3.0}, {9,9.0}, {6,6.0}})
            : ((const struct mtxfile_vector_coordinate_real_double[2]) {
                    {1,1.0}, {8,8.0}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {4,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(5, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(3, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(3, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(0, dsts[0].partition.part_sizes[1]);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[0].data.vector_coordinate_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(4.0, data[0].a);
                TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(3.0, data[1].a);
                TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(1.0, data[2].a);
            } else if (rank == 1) {
                /* empty */
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(4, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(3, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[1].partition.part_sizes[1]);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[1].data.vector_coordinate_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(9.0, data[0].a);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(6.0, data[0].a);
                TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(8.0, data[1].a);
            }
            mtxdistfile_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    {
        int num_rows = 9;
        int64_t num_nonzeros = 6;
        const struct mtxfile_vector_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_double[4]) {
                    {4,4.0}, {3,3.0}, {9,9.0}, {6,6.0}})
            : ((const struct mtxfile_vector_coordinate_real_double[2]) {
                    {1,1.0}, {8,8.0}});

        const int num_parts = comm_size;
        int64_t part_sizes[] = {4,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        const int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_cyclic;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile dsts[num_row_parts];
        err = mtxdistfile_partition(dsts, &src, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(5, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(3, dsts[0].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[0].partition.size);
            TEST_ASSERT_EQ(2, dsts[0].partition.num_parts);
            TEST_ASSERT_EQ(3, dsts[0].partition.part_sizes[0]);
            TEST_ASSERT_EQ(0, dsts[0].partition.part_sizes[1]);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[0].data.vector_coordinate_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(3.0, data[0].a);
                TEST_ASSERT_EQ(5, data[1].i); TEST_ASSERT_EQ(9.0, data[1].a);
                TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(1.0, data[2].a);
            } else if (rank == 1) {
                /* empty */
            }
            mtxdistfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(4, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(3, dsts[1].size.num_nonzeros);
            TEST_ASSERT_EQ(3, dsts[1].partition.size);
            TEST_ASSERT_EQ(2, dsts[1].partition.num_parts);
            TEST_ASSERT_EQ(1, dsts[1].partition.part_sizes[0]);
            TEST_ASSERT_EQ(2, dsts[1].partition.part_sizes[1]);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[1].data.vector_coordinate_real_double;
            if (rank == 0) {
                TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(4.0, data[0].a);
            } else if (rank == 1) {
                TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(6.0, data[0].a);
                TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(8.0, data[1].a);
            }
            mtxdistfile_free(&dsts[1]);
        }
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistfile_join()’ tests joining distributed Matrix Market
 * files.
 */
int test_mtxdistfile_join(void)
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
#if 0
    {
        int num_rows = 3;
        int num_columns = 3;
        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 6;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {4,2};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 3;
        int64_t data_part_sizes1[] = {1,2};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxdistfile srcs[num_parts];

        const double * srcdata0 = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 3.0, 4.0})
            : ((const double[2]) {5.0, 6.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[0], mtxfile_general, rowpart.part_sizes[0], num_columns, srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[1]) {7.0})
            : ((const double[2]) {8.0, 9.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[1], mtxfile_general, rowpart.part_sizes[1], num_columns, srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, data[0]);
            TEST_ASSERT_EQ(7.0, data[1]);
            TEST_ASSERT_EQ(8.0, data[2]);
            TEST_ASSERT_EQ(9.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 9;
        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_cyclic;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 6;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {4,2};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 3;
        int64_t data_part_sizes1[] = {1,2};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxdistfile srcs[num_parts];
        const double * srcdata0 = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 3.0, 7.0})
            : ((const double[2]) {8.0, 9.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[0], mtxfile_general, rowpart.part_sizes[0], num_columns, srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[1]) {4.0})
            : ((const double[2]) {5.0, 6.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[1], mtxfile_general, rowpart.part_sizes[1], num_columns, srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, data[0]);
            TEST_ASSERT_EQ(7.0, data[1]);
            TEST_ASSERT_EQ(8.0, data[2]);
            TEST_ASSERT_EQ(9.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 9;
        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int num_col_parts = 1;
        enum mtxpartitioning colpart_type = mtx_block;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 6;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {4,2};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 3;
        int64_t data_part_sizes1[] = {1,2};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxdistfile srcs[num_parts];
        const double * srcdata0 = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 3.0, 4.0})
            : ((const double[2]) {5.0, 6.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[0], mtxfile_general, rowpart.part_sizes[0], num_columns, srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[1]) {7.0})
            : ((const double[2]) {8.0, 9.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[1], mtxfile_general, rowpart.part_sizes[1], num_columns, srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, data[0]);
            TEST_ASSERT_EQ(7.0, data[1]);
            TEST_ASSERT_EQ(8.0, data[2]);
            TEST_ASSERT_EQ(9.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 9;
        int num_col_parts = 2;
        enum mtxpartitioning colpart_type = mtx_block;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 6;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {4,2};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 3;
        int64_t data_part_sizes1[] = {1,2};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_col_parts;
        struct mtxdistfile srcs[num_parts];
        const double * srcdata0 = (rank == 0)
            ? ((const double[4]) {1.0, 2.0, 4.0, 5.0})
            : ((const double[2]) {7.0, 8.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[0], mtxfile_general, num_rows, colpart.part_sizes[0], srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[1]) {3.0})
            : ((const double[2]) {6.0, 9.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[1], mtxfile_general, num_rows, colpart.part_sizes[1], srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, NULL, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, data[0]);
            TEST_ASSERT_EQ(7.0, data[1]);
            TEST_ASSERT_EQ(8.0, data[2]);
            TEST_ASSERT_EQ(9.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxpartition_free(&colpart);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 9;
        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int num_col_parts = 2;
        enum mtxpartitioning colpart_type = mtx_block;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 4;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {2,2};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 2;
        int64_t data_part_sizes1[] = {2,0};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size2 = 2;
        int64_t data_part_sizes2[] = {1,1};
        struct mtxpartition datapart2;
        err = mtxpartition_init_block(
            &datapart2, size2, num_data_parts, data_part_sizes2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size3 = 1;
        int64_t data_part_sizes3[] = {0,1};
        struct mtxpartition datapart3;
        err = mtxpartition_init_block(
            &datapart3, size3, num_data_parts, data_part_sizes3);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts * num_col_parts;
        struct mtxdistfile srcs[num_parts];
        const double * srcdata0 = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[2]) {4.0, 5.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[0], mtxfile_general,
            rowpart.part_sizes[0], colpart.part_sizes[0], srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[2]) {3.0, 6.0})
            : NULL;
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[1], mtxfile_general,
            rowpart.part_sizes[0], colpart.part_sizes[1], srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);
        const double * srcdata2 = (rank == 0)
            ? ((const double[1]) {7.0})
            : ((const double[1]) {8.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[2], mtxfile_general,
            rowpart.part_sizes[1], colpart.part_sizes[0], srcdata2,
            &datapart2, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart2);
        const double * srcdata3 = (rank == 0)
            ? NULL
            : ((const double[1]) {9.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[3], mtxfile_general,
            rowpart.part_sizes[1], colpart.part_sizes[1], srcdata3,
            &datapart3, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart3);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

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
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, data[0]);
            TEST_ASSERT_EQ(7.0, data[1]);
            TEST_ASSERT_EQ(8.0, data[2]);
            TEST_ASSERT_EQ(9.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&srcs[3]);
        mtxdistfile_free(&srcs[2]);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
        mtxpartition_free(&rowpart);
        mtxpartition_free(&colpart);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        int64_t size = 9;
        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_cyclic;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int num_col_parts = 2;
        enum mtxpartitioning colpart_type = mtx_block;
        struct mtxpartition colpart;
        err = mtxpartition_init(
            &colpart, colpart_type, num_columns, num_col_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 4;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {2,2};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 2;
        int64_t data_part_sizes1[] = {2,0};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size2 = 2;
        int64_t data_part_sizes2[] = {1,1};
        struct mtxpartition datapart2;
        err = mtxpartition_init_block(
            &datapart2, size2, num_data_parts, data_part_sizes2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size3 = 1;
        int64_t data_part_sizes3[] = {0,1};
        struct mtxpartition datapart3;
        err = mtxpartition_init_block(
            &datapart3, size3, num_data_parts, data_part_sizes3);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts * num_col_parts;
        struct mtxdistfile srcs[num_parts];
        const double * srcdata0 = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[2]) {7.0, 8.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[0], mtxfile_general,
            rowpart.part_sizes[0], colpart.part_sizes[0], srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[2]) {3.0, 9.0})
            : NULL;
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[1], mtxfile_general,
            rowpart.part_sizes[0], colpart.part_sizes[1], srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);
        const double * srcdata2 = (rank == 0)
            ? ((const double[1]) {4.0})
            : ((const double[1]) {5.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[2], mtxfile_general,
            rowpart.part_sizes[1], colpart.part_sizes[0], srcdata2,
            &datapart2, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart2);
        const double * srcdata3 = (rank == 0)
            ? NULL
            : ((const double[1]) {6.0});
        err = mtxdistfile_init_matrix_array_real_double(
            &srcs[3], mtxfile_general,
            rowpart.part_sizes[1], colpart.part_sizes[1], srcdata3,
            &datapart3, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart3);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, &colpart, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

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
        TEST_ASSERT_EQ(5, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(6.0, data[0]);
            TEST_ASSERT_EQ(7.0, data[1]);
            TEST_ASSERT_EQ(8.0, data[2]);
            TEST_ASSERT_EQ(9.0, data[3]);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&srcs[3]);
        mtxdistfile_free(&srcs[2]);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
        mtxpartition_free(&rowpart);
        mtxpartition_free(&colpart);
    }

    {
        int num_rows = 8;
        int64_t size = 8;
        int num_row_parts = 3;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 3;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {2,1};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 3;
        int64_t data_part_sizes1[] = {1,2};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size2 = 2;
        int64_t data_part_sizes2[] = {1,1};
        struct mtxpartition datapart2;
        err = mtxpartition_init_block(
            &datapart2, size2, num_data_parts, data_part_sizes2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxdistfile srcs[num_parts];
        const double * srcdata0 = (rank == 0)
            ? ((const double[2]) {1.0, 2.0})
            : ((const double[2]) {3.0});
        err = mtxdistfile_init_vector_array_real_double(
            &srcs[0], rowpart.part_sizes[0], srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const double * srcdata1 = (rank == 0)
            ? ((const double[1]) {4.0})
            : ((const double[2]) {5.0, 6.0});
        err = mtxdistfile_init_vector_array_real_double(
            &srcs[1], rowpart.part_sizes[1], srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);
        const double * srcdata2 = (rank == 0)
            ? ((const double[1]) {7.0})
            : ((const double[1]) {8.0});
        err = mtxdistfile_init_vector_array_real_double(
            &srcs[2], rowpart.part_sizes[2], srcdata2,
            &datapart2, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart2);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
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
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(4, dst.partition.part_sizes[1]);
        const double * data = dst.data.array_real_double;
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
        mtxdistfile_free(&srcs[2]);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
        mtxpartition_free(&rowpart);
    }
#endif
    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        int num_row_parts = 2;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 2;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {1,1};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 2;
        int64_t data_part_sizes1[] = {1,1};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxdistfile srcs[num_parts];
        const struct mtxfile_matrix_coordinate_real_double * srcdata0 = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[1])
                {2,2,5.0})
            : ((const struct mtxfile_matrix_coordinate_real_double[1])
                {1,1,1.0});
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &srcs[0], mtxfile_general, rowpart.part_sizes[0],
            num_columns, size0, srcdata0, &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const struct mtxfile_matrix_coordinate_real_double * srcdata1 = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[1])
                {1,3,9.0})
            : ((const struct mtxfile_matrix_coordinate_real_double[1])
                {1,2,8.0});
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &srcs[1], mtxfile_general, rowpart.part_sizes[1],
            num_columns, size1, srcdata1, &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(3, dst.size.num_rows);
        TEST_ASSERT_EQ(3, dst.size.num_columns);
        TEST_ASSERT_EQ(4, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(4, dst.partition.size);
        TEST_ASSERT_EQ(2, dst.partition.num_parts);
        TEST_ASSERT_EQ(2, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, dst.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            dst.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(2, data[0].j);
            TEST_ASSERT_EQ(5.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(3, data[1].j);
            TEST_ASSERT_EQ(9.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(8.0, data[1].a);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
        mtxpartition_free(&rowpart);
    }

    {
        int num_rows = 8;
        int num_row_parts = 3;
        enum mtxpartitioning rowpart_type = mtx_block;
        struct mtxpartition rowpart;
        err = mtxpartition_init(
            &rowpart, rowpart_type, num_rows, num_row_parts, NULL, 0, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t size0 = 2;
        const int num_data_parts = comm_size;
        int64_t data_part_sizes0[] = {1,1};
        struct mtxpartition datapart0;
        err = mtxpartition_init_block(
            &datapart0, size0, num_data_parts, data_part_sizes0);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size1 = 2;
        int64_t data_part_sizes1[] = {1,1};
        struct mtxpartition datapart1;
        err = mtxpartition_init_block(
            &datapart1, size1, num_data_parts, data_part_sizes1);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t size2 = 1;
        int64_t data_part_sizes2[] = {0,1};
        struct mtxpartition datapart2;
        err = mtxpartition_init_block(
            &datapart2, size2, num_data_parts, data_part_sizes2);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = num_row_parts;
        struct mtxdistfile srcs[num_parts];
        const struct mtxfile_vector_coordinate_real_double * srcdata0 = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_double[1])
                {1,1.0})
            : ((const struct mtxfile_vector_coordinate_real_double[1])
                {2,2.0});
        err = mtxdistfile_init_vector_coordinate_real_double(
            &srcs[0], rowpart.part_sizes[0], size0, srcdata0,
            &datapart0, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart0);
        const struct mtxfile_vector_coordinate_real_double * srcdata1 = (rank == 0)
            ? ((const struct mtxfile_vector_coordinate_real_double[1])
                {1,4.0})
            : ((const struct mtxfile_vector_coordinate_real_double[1])
                {2,5.0});
        err = mtxdistfile_init_vector_coordinate_real_double(
            &srcs[1], rowpart.part_sizes[1], size1, srcdata1,
            &datapart1, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart1);
        const struct mtxfile_vector_coordinate_real_double * srcdata2 = (rank == 0)
            ? NULL
            : ((const struct mtxfile_vector_coordinate_real_double[1])
                {2,8.0});
        err = mtxdistfile_init_vector_coordinate_real_double(
            &srcs[2], rowpart.part_sizes[2], size2, srcdata2,
            &datapart2, comm, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        mtxpartition_free(&datapart2);

        struct mtxdistfile dst;
        err = mtxdistfile_join(&dst, srcs, &rowpart, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(8, dst.size.num_rows);
        TEST_ASSERT_EQ(-1, dst.size.num_columns);
        TEST_ASSERT_EQ(5, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(5, dst.partition.size);
        TEST_ASSERT_EQ(2, dst.partition.num_parts);
        TEST_ASSERT_EQ(2, dst.partition.part_sizes[0]);
        TEST_ASSERT_EQ(3, dst.partition.part_sizes[1]);
        const struct mtxfile_vector_coordinate_real_double * data =
            dst.data.vector_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(4.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(2.0, data[0].a);
            TEST_ASSERT_EQ(5, data[1].i); TEST_ASSERT_EQ(5.0, data[1].a);
            TEST_ASSERT_EQ(8, data[2].i); TEST_ASSERT_EQ(8.0, data[2].a);
        }
        mtxdistfile_free(&dst);
        mtxdistfile_free(&srcs[2]);
        mtxdistfile_free(&srcs[1]);
        mtxdistfile_free(&srcs[0]);
        mtxpartition_free(&rowpart);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistfile_transpose()' tests transposing distributed
 * matrices.
 */
int test_mtxdistfile_transpose(void)
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
     * Matrix array formats.
     */

    {
        int num_rows = 3;
        int num_columns = 4;
        int num_local_rows = rank == 0 ? 2 : 1;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[1]) {2});
        int num_local_columns = rank == 0 ? 4 : 4;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[4]) {0, 1, 2, 3})
            : ((const int64_t[4]) {0, 1, 2, 3});
        const double * srcdata = (rank == 0)
            ? ((const double[8]) {1.0,  2.0,  3.0,  4.0, 5.0, 6.0, 7.0, 8.0})
            : ((const double[4]) {9.0, 10.0, 11.0, 12.0});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistfile_transpose(&src, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_array, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(4, src.size.num_rows);
        TEST_ASSERT_EQ(3, src.size.num_columns);
        TEST_ASSERT_EQ(-1, src.size.num_nonzeros);
        TEST_ASSERT_EQ(12, src.partition.size);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(6, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(6, src.partition.part_sizes[1]);
        const double * data = src.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ( 1.0, data[0]);
            TEST_ASSERT_EQ( 5.0, data[1]);
            TEST_ASSERT_EQ( 9.0, data[2]);
            TEST_ASSERT_EQ( 2.0, data[3]);
            TEST_ASSERT_EQ( 6.0, data[4]);
            TEST_ASSERT_EQ(10.0, data[5]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 3.0, data[0]);
            TEST_ASSERT_EQ( 7.0, data[1]);
            TEST_ASSERT_EQ(11.0, data[2]);
            TEST_ASSERT_EQ( 4.0, data[3]);
            TEST_ASSERT_EQ( 8.0, data[4]);
            TEST_ASSERT_EQ(12.0, data[5]);
        }
        mtxdistfile_free(&src);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 4;
        int num_columns = 5;
        int64_t num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {1,2,2.0}, {1,3,3.0}, {2,3,7.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {1,4,4.0}, {3,4,11.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistfile_transpose(&src, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(5, src.size.num_rows);
        TEST_ASSERT_EQ(4, src.size.num_columns);
        TEST_ASSERT_EQ(5, src.size.num_nonzeros);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(3, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, src.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            src.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(2.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(1, data[1].j);
            TEST_ASSERT_EQ(3.0, data[1].a);
            TEST_ASSERT_EQ(3, data[2].i); TEST_ASSERT_EQ(2, data[2].j);
            TEST_ASSERT_EQ(7.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(4.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(3, data[1].j);
            TEST_ASSERT_EQ(11.0, data[1].a);
        }
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * `test_mtxdistfile_sort()' tests sorting Matrix Market files.
 */
int test_mtxdistfile_sort(void)
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
     * Matrix array formats.
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        int num_local_rows = rank == 0 ? 2 : 2;
        const int64_t * rowmap = rank == 0
            ? ((const int64_t[2]) {0, 1})
            : ((const int64_t[2]) {2, 3});
        int num_local_columns = rank == 0 ? 4 : 4;
        const int64_t * colmap = rank == 0
            ? ((const int64_t[4]) {0, 1, 2, 3})
            : ((const int64_t[4]) {0, 1, 2, 3});
        const double * srcdata = (rank == 0)
            ? ((const double[8]) {1, 2, 3, 4, 5, 6, 7, 8})
            : ((const double[8]) {9,10,11,12,13,14,15,16});

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            num_local_rows, rowmap, num_local_columns, colmap, srcdata,
            comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistfile_sort(
            &src, mtxfile_column_major, 0, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_array, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(4, src.size.num_rows);
        TEST_ASSERT_EQ(4, src.size.num_columns);
        TEST_ASSERT_EQ(-1, src.size.num_nonzeros);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(8, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(8, src.partition.part_sizes[1]);
        const double * data = src.data.array_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ( 1.0, data[0]);
            TEST_ASSERT_EQ( 5.0, data[1]);
            TEST_ASSERT_EQ( 9.0, data[2]);
            TEST_ASSERT_EQ(13.0, data[3]);
            TEST_ASSERT_EQ( 2.0, data[4]);
            TEST_ASSERT_EQ( 6.0, data[5]);
            TEST_ASSERT_EQ(10.0, data[6]);
            TEST_ASSERT_EQ(14.0, data[7]);
        } else if (rank == 1) {
            TEST_ASSERT_EQ( 3.0, data[0]);
            TEST_ASSERT_EQ( 7.0, data[1]);
            TEST_ASSERT_EQ(11.0, data[2]);
            TEST_ASSERT_EQ(15.0, data[3]);
            TEST_ASSERT_EQ( 4.0, data[4]);
            TEST_ASSERT_EQ( 8.0, data[5]);
            TEST_ASSERT_EQ(12.0, data[6]);
            TEST_ASSERT_EQ(16.0, data[7]);
        }
        mtxdistfile_free(&src);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        int64_t num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {4,4,4.0}, {1,1,1.0}, {3,3,3.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {4,1,4.1}, {2,2,2.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        int64_t * perm = (rank == 0)
            ? ((int64_t[3]) {3, 4, 1})
            : ((int64_t[2]) {2, 5});
        err = mtxdistfile_sort(
            &src, mtxfile_permutation, rank == 0 ? 3 : 2, perm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(4, src.size.num_rows);
        TEST_ASSERT_EQ(4, src.size.num_columns);
        TEST_ASSERT_EQ(5, src.size.num_nonzeros);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(3, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, src.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            src.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
            TEST_ASSERT_EQ(3.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(1, data[1].j);
            TEST_ASSERT_EQ(4.1, data[1].a);
            TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(4, data[2].j);
            TEST_ASSERT_EQ(4.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
        }
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        int64_t num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {4,4,4.0}, {1,1,1.0}, {3,3,3.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {4,1,4.1}, {2,2,2.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistfile_sort(
            &src, mtxfile_row_major, 0, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(4, src.size.num_rows);
        TEST_ASSERT_EQ(4, src.size.num_columns);
        TEST_ASSERT_EQ(5, src.size.num_nonzeros);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(3, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, src.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            src.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(3, data[2].i); TEST_ASSERT_EQ(3, data[2].j);
            TEST_ASSERT_EQ(3.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(4.1, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(4, data[1].j);
            TEST_ASSERT_EQ(4.0, data[1].a);
        }
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        int64_t num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {4,4,4.0}, {1,1,1.0}, {3,3,3.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {4,1,4.1}, {2,2,2.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistfile_sort(
            &src, mtxfile_column_major, 0, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(4, src.size.num_rows);
        TEST_ASSERT_EQ(4, src.size.num_columns);
        TEST_ASSERT_EQ(5, src.size.num_nonzeros);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(3, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, src.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            src.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(1, data[1].j);
            TEST_ASSERT_EQ(4.1, data[1].a);
            TEST_ASSERT_EQ(2, data[2].i); TEST_ASSERT_EQ(2, data[2].j);
            TEST_ASSERT_EQ(2.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
            TEST_ASSERT_EQ(3.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(4, data[1].j);
            TEST_ASSERT_EQ(4.0, data[1].a);
        }
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        int64_t num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = (rank == 0)
            ? ((const struct mtxfile_matrix_coordinate_real_double[3]) {
                    {4,4,4.0}, {1,1,1.0}, {3,3,3.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2]) {
                    {4,1,4.1}, {2,2,2.0}});

        int num_parts = comm_size;
        int64_t part_sizes[] = {3,2};
        struct mtxpartition partition;
        err = mtxpartition_init_block(
            &partition, num_nonzeros, num_parts, part_sizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata,
            &partition, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        err = mtxdistfile_sort(
            &src, mtxfile_morton, 0, NULL, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        TEST_ASSERT_EQ(mtxfile_matrix, src.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, src.header.format);
        TEST_ASSERT_EQ(mtxfile_real, src.header.field);
        TEST_ASSERT_EQ(mtxfile_general, src.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, src.precision);
        TEST_ASSERT_EQ(4, src.size.num_rows);
        TEST_ASSERT_EQ(4, src.size.num_columns);
        TEST_ASSERT_EQ(5, src.size.num_nonzeros);
        TEST_ASSERT_EQ(2, src.partition.num_parts);
        TEST_ASSERT_EQ(3, src.partition.part_sizes[0]);
        TEST_ASSERT_EQ(2, src.partition.part_sizes[1]);
        const struct mtxfile_matrix_coordinate_real_double * data =
            src.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
            TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(1, data[2].j);
            TEST_ASSERT_EQ(4.1, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
            TEST_ASSERT_EQ(3.0, data[0].a);
            TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(4, data[1].j);
            TEST_ASSERT_EQ(4.0, data[1].a);
        }
        mtxdistfile_free(&src);
        mtxpartition_free(&partition);
    }
    mtxdisterror_free(&disterr);
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
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }
    mpierr = MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* 2. Run test suite. */
    TEST_SUITE_BEGIN("Running tests for distributed Matrix Market files\n");
    TEST_RUN(test_mtxdistfile_from_mtxfile);
    TEST_RUN(test_mtxdistfile_fread_shared);
    TEST_RUN(test_mtxdistfile_fwrite_shared);
    TEST_RUN(test_mtxdistfile_partition);
    TEST_RUN(test_mtxdistfile_join);
    TEST_RUN(test_mtxdistfile_transpose);
    TEST_RUN(test_mtxdistfile_sort);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
