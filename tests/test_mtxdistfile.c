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
 * Last modified: 2022-05-01
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
 * ‘test_mtxdistfile_from_mtxfile_rowwise()’ tests converting Matrix
 * Market files stored on a single process to distributed Matrix
 * Market files based on a rowwise partitioning.
 */
int test_mtxdistfile_from_mtxfile_rowwise(void)
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
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t partsize = rank == 0 ? 2 : 1;
        struct mtxdistfile dst;
        err = mtxdistfile_from_mtxfile_rowwise(
            &dst, &src, mtx_block, partsize, 0, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(3, dst.size.num_rows);
        TEST_ASSERT_EQ(3, dst.size.num_columns);
        TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(9, dst.datasize);
        union mtxfiledata * data = &dst.data;
        if (rank == 0) {
            TEST_ASSERT_EQ(6, dst.localdatasize);
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
            TEST_ASSERT_EQ(data->array_real_double[4], 5.0);
            TEST_ASSERT_EQ(data->array_real_double[5], 6.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(3, dst.localdatasize);
            TEST_ASSERT_EQ(data->array_real_double[0], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 8.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 9.0);
        }
        mtxdistfile_free(&dst);
        mtxfile_free(&src);
    }

    {
        int num_rows = 8;
        const double mtxdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile src;
        err = mtxfile_init_vector_array_real_double(
            &src, num_rows, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t partsize = rank == 0 ? 4 : 4;
        struct mtxdistfile dst;
        err = mtxdistfile_from_mtxfile_rowwise(
            &dst, &src, mtx_block, partsize, 0, comm, root, &disterr);
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
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(8, dst.datasize);
        union mtxfiledata * data = &dst.data;
        if (rank == 0) {
            TEST_ASSERT_EQ(4, dst.localdatasize);
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(4, dst.localdatasize);
            TEST_ASSERT_EQ(data->array_real_double[0], 5.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 6.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 8.0);
        }
        mtxdistfile_free(&dst);
        mtxfile_free(&src);
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
        struct mtxfile src;
        err = mtxfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general,
            num_rows, num_columns, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t partsize = rank == 0 ? 2 : 3;
        struct mtxdistfile dst;
        err = mtxdistfile_from_mtxfile_rowwise(
            &dst, &src, mtx_block, partsize, 0, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(5, dst.size.num_rows);
        TEST_ASSERT_EQ(5, dst.size.num_columns);
        TEST_ASSERT_EQ(4, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(4, dst.datasize);
        const struct mtxfile_matrix_coordinate_real_double * data =
            dst.data.matrix_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(2, dst.localdatasize);
            TEST_ASSERT_EQ(  2, data[0].i); TEST_ASSERT_EQ(   2, data[0].j);
            TEST_ASSERT_EQ(2.0, data[0].a);
            TEST_ASSERT_EQ(  1, data[1].i); TEST_ASSERT_EQ(   1, data[1].j);
            TEST_ASSERT_EQ(1.0, data[1].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(2, dst.localdatasize);
            TEST_ASSERT_EQ(  4, data[0].i); TEST_ASSERT_EQ(   4, data[0].j);
            TEST_ASSERT_EQ(4.0, data[0].a);
            TEST_ASSERT_EQ(  3, data[1].i); TEST_ASSERT_EQ(   3, data[1].j);
            TEST_ASSERT_EQ(3.0, data[1].a);
        }
        mtxdistfile_free(&dst);
        mtxfile_free(&src);
    }

    /*
     * Vector coordinate formats
     */

    {
        int num_rows = 6;
        const struct mtxfile_vector_coordinate_real_double mtxdata[] = {
            {5, 5.0}, {3, 3.0}, {2, 2.0}, {1, 1.0}};
        int64_t num_nonzeros = sizeof(mtxdata) / sizeof(*mtxdata);
        struct mtxfile src;
        err = mtxfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, mtxdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        int64_t partsize = rank == 0 ? 3 : 3;
        struct mtxdistfile dst;
        err = mtxdistfile_from_mtxfile_rowwise(
            &dst, &src, mtx_block, partsize, 0, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
        TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
        TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(6, dst.size.num_rows);
        TEST_ASSERT_EQ(-1, dst.size.num_columns);
        TEST_ASSERT_EQ(4, dst.size.num_nonzeros);
        TEST_ASSERT_EQ(mtx_double, dst.precision);
        TEST_ASSERT_EQ(4, dst.datasize);
        const struct mtxfile_vector_coordinate_real_double * data =
            dst.data.vector_coordinate_real_double;
        if (rank == 0) {
            TEST_ASSERT_EQ(3, dst.localdatasize);
            TEST_ASSERT_EQ(1, dst.idx[0]);
            TEST_ASSERT_EQ(2, dst.idx[1]);
            TEST_ASSERT_EQ(3, dst.idx[2]);
            TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(3.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2.0, data[1].a);
            TEST_ASSERT_EQ(1, data[2].i); TEST_ASSERT_EQ(1.0, data[2].a);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, dst.localdatasize);
            TEST_ASSERT_EQ(0, dst.idx[0]);
            TEST_ASSERT_EQ(5, data[0].i); TEST_ASSERT_EQ(5.0, data[0].a);
        }
        mtxdistfile_free(&dst);
        mtxfile_free(&src);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistfile_fread_rowwise()’ tests reading Matrix Market
 * files from a stream and distributing them among MPI processes.
 */
int test_mtxdistfile_fread_rowwise(void)
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL,
            comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.array_real_single[0], 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.array_real_double[0], 1.5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_single[0][0], 1.5f);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_single[0][1], 2.1f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_double[0][0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.array_complex_double[0][1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.array_integer_single[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.array_integer_double[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_pattern[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.matrix_coordinate_pattern[0].j, 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
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
        err = mtxdistfile_fread_rowwise(
            &mtxdistfile, precision, mtx_block_cyclic, 0, 1,
            f, &lines_read, &bytes_read, 0, NULL, comm, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s",
            err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr)
            : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ_MSG(
                strlen(s), bytes_read,
                "read %"PRId64" of %zu bytes", bytes_read, strlen(s));
            TEST_ASSERT_EQ(5, lines_read);
        }
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile.precision);
        TEST_ASSERT_EQ(2, mtxdistfile.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_pattern[0].i, 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile.data.vector_coordinate_pattern[0].i, 2);
        }
        mtxdistfile_free(&mtxdistfile);
        fclose(f);
    }
    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistfile_to_mtxfile()’ tests converting to Matrix Market
 * files stored on a single process from distributed Matrix Market
 * files.
 */
int test_mtxdistfile_to_mtxfile(void)
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
    if (err) MPI_Abort(comm, EXIT_FAILURE);

    /*
     * array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});
        int64_t localsize = rank == 0 ? 6 : 3;

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            localsize, NULL, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxdistfile_to_mtxfile(&dst, &src, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(3, dst.size.num_rows);
            TEST_ASSERT_EQ(3, dst.size.num_columns);
            TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(9, dst.datasize);
            union mtxfiledata * data = &dst.data;
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
            TEST_ASSERT_EQ(data->array_real_double[4], 5.0);
            TEST_ASSERT_EQ(data->array_real_double[5], 6.0);
            TEST_ASSERT_EQ(data->array_real_double[6], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[7], 8.0);
            TEST_ASSERT_EQ(data->array_real_double[8], 9.0);
            mtxfile_free(&dst);
        }
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const int64_t * idx = (rank == 0)
            ? ((const int64_t[6]) {0, 1, 2, 6, 7, 8})
            : ((const int64_t[3]) {3, 4, 5});
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 7.0, 8.0, 9.0})
            : ((const double[3]) {4.0, 5.0, 6.0});
        int64_t localsize = rank == 0 ? 6 : 3;

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns,
            localsize, idx, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxdistfile_to_mtxfile(&dst, &src, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(3, dst.size.num_rows);
            TEST_ASSERT_EQ(3, dst.size.num_columns);
            TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(9, dst.datasize);
            union mtxfiledata * data = &dst.data;
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
            TEST_ASSERT_EQ(data->array_real_double[4], 5.0);
            TEST_ASSERT_EQ(data->array_real_double[5], 6.0);
            TEST_ASSERT_EQ(data->array_real_double[6], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[7], 8.0);
            TEST_ASSERT_EQ(data->array_real_double[8], 9.0);
            mtxfile_free(&dst);
        }
        mtxdistfile_free(&src);
    }

    {
        int num_rows = 9;
        const double * srcdata = (rank == 0)
            ? ((const double[6]) {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
            : ((const double[3]) {7.0, 8.0, 9.0});
        int64_t localsize = rank == 0 ? 6 : 3;

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_array_real_double(
            &src, num_rows, localsize, NULL, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxdistfile_to_mtxfile(&dst, &src, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_array, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(9, dst.size.num_rows);
            TEST_ASSERT_EQ(-1, dst.size.num_columns);
            TEST_ASSERT_EQ(-1, dst.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(9, dst.datasize);
            union mtxfiledata * data = &dst.data;
            TEST_ASSERT_EQ(data->array_real_double[0], 1.0);
            TEST_ASSERT_EQ(data->array_real_double[1], 2.0);
            TEST_ASSERT_EQ(data->array_real_double[2], 3.0);
            TEST_ASSERT_EQ(data->array_real_double[3], 4.0);
            TEST_ASSERT_EQ(data->array_real_double[4], 5.0);
            TEST_ASSERT_EQ(data->array_real_double[5], 6.0);
            TEST_ASSERT_EQ(data->array_real_double[6], 7.0);
            TEST_ASSERT_EQ(data->array_real_double[7], 8.0);
            TEST_ASSERT_EQ(data->array_real_double[8], 9.0);
            mtxfile_free(&dst);
        }
        mtxdistfile_free(&src);
    }

    /*
     * matrix coordinate formats
     */

    {
        int num_rows = 6;
        int num_columns = 4;
        int num_nonzeros = 5;
        const struct mtxfile_matrix_coordinate_real_double * srcdata = rank == 0
            ? ((const struct mtxfile_matrix_coordinate_real_double[3])
                {{4, 4, 16.0}, {3, 3, 9.0}, {2, 2, 4.0}})
            : ((const struct mtxfile_matrix_coordinate_real_double[2])
                {{1, 1, 1.0}, {1, 3, 3.0}});
        int64_t localsize = rank == 0 ? 3 : 2;

        struct mtxdistfile src;
        err = mtxdistfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros,
            localsize, NULL, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxdistfile_to_mtxfile(&dst, &src, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_matrix, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(6, dst.size.num_rows);
            TEST_ASSERT_EQ(4, dst.size.num_columns);
            TEST_ASSERT_EQ(5, dst.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(5, dst.datasize);
            const struct mtxfile_matrix_coordinate_real_double * data =
                dst.data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(4, data[0].j);
            TEST_ASSERT_EQ(16.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(3, data[1].j);
            TEST_ASSERT_EQ( 9.0, data[1].a);
            TEST_ASSERT_EQ(2, data[2].i); TEST_ASSERT_EQ(2, data[2].j);
            TEST_ASSERT_EQ( 4.0, data[2].a);
            TEST_ASSERT_EQ(1, data[3].i); TEST_ASSERT_EQ(1, data[3].j);
            TEST_ASSERT_EQ( 1.0, data[3].a);
            TEST_ASSERT_EQ(1, data[4].i); TEST_ASSERT_EQ(3, data[4].j);
            TEST_ASSERT_EQ( 3.0, data[4].a);
            mtxfile_free(&dst);
        }
        mtxdistfile_free(&src);
    }

    /*
     * vector coordinate formats
     */

    {
        int num_rows = 6;
        int num_nonzeros = 5;
        const struct mtxfile_vector_coordinate_real_double * srcdata = rank == 0
            ? ((const struct mtxfile_vector_coordinate_real_double[3])
                {{4, 16.0}, {3, 9.0}, {2, 4.0}})
            : ((const struct mtxfile_vector_coordinate_real_double[2])
                {{1, 1.0}, {5, 3.0}});
        int64_t localsize = rank == 0 ? 3 : 2;

        struct mtxdistfile src;
        err = mtxdistfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, localsize, NULL, srcdata, comm, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));

        struct mtxfile dst;
        err = mtxdistfile_to_mtxfile(&dst, &src, root, &disterr);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%s", err == MTX_ERR_MPI_COLLECTIVE
            ? mtxdisterror_description(&disterr) : mtxstrerror(err));
        if (rank == root) {
            TEST_ASSERT_EQ(mtxfile_vector, dst.header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dst.header.format);
            TEST_ASSERT_EQ(mtxfile_real, dst.header.field);
            TEST_ASSERT_EQ(mtxfile_general, dst.header.symmetry);
            TEST_ASSERT_EQ(6, dst.size.num_rows);
            TEST_ASSERT_EQ(-1, dst.size.num_columns);
            TEST_ASSERT_EQ(5, dst.size.num_nonzeros);
            TEST_ASSERT_EQ(mtx_double, dst.precision);
            TEST_ASSERT_EQ(5, dst.datasize);
            const struct mtxfile_vector_coordinate_real_double * data =
                dst.data.vector_coordinate_real_double;
            TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(16.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ( 9.0, data[1].a);
            TEST_ASSERT_EQ(2, data[2].i); TEST_ASSERT_EQ( 4.0, data[2].a);
            TEST_ASSERT_EQ(1, data[3].i); TEST_ASSERT_EQ( 1.0, data[3].a);
            TEST_ASSERT_EQ(5, data[4].i); TEST_ASSERT_EQ( 3.0, data[4].a);
            mtxfile_free(&dst);
        }
        mtxdistfile_free(&src);
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
    TEST_RUN(test_mtxdistfile_from_mtxfile_rowwise);
    TEST_RUN(test_mtxdistfile_fread_rowwise);
    TEST_RUN(test_mtxdistfile_to_mtxfile);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
