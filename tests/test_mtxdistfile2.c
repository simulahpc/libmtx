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
#include <libmtx/mtxfile/mtxdistfile2.h>
#include <libmtx/util/partition.h>

#include <mpi.h>

#include <errno.h>

#include <stdlib.h>

const char * program_invocation_short_name = "test_mtxdistfile2";

/**
 * ‘test_mtxdistfile2_from_mtxfile_rowwise()’ tests converting Matrix
 * Market files stored on a single process to distributed Matrix
 * Market files based on a rowwise partitioning.
 */
int test_mtxdistfile2_from_mtxfile_rowwise(void)
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
    if (err)MPI_Abort(comm, EXIT_FAILURE);

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
        struct mtxdistfile2 dst;
        err = mtxdistfile2_from_mtxfile_rowwise(
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
        mtxdistfile2_free(&dst);
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
        struct mtxdistfile2 dst;
        err = mtxdistfile2_from_mtxfile_rowwise(
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
        mtxdistfile2_free(&dst);
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
        struct mtxdistfile2 dst;
        err = mtxdistfile2_from_mtxfile_rowwise(
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
        mtxdistfile2_free(&dst);
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
        struct mtxdistfile2 dst;
        err = mtxdistfile2_from_mtxfile_rowwise(
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
        mtxdistfile2_free(&dst);
        mtxfile_free(&src);
    }

    mtxdisterror_free(&disterr);
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxdistfile2_fread_rowwise()’ tests reading Matrix Market
 * files from a stream and distributing them among MPI processes.
 */
int test_mtxdistfile2_fread_rowwise(void)
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_real_single[0], 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_real_single[0], 1.6f);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_real_double[0], 1.5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_real_double[0], 1.6);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_single[0][0], 1.5f);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_single[0][1], 2.1f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_single[0][0], 1.6f);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_single[0][1], 2.2f);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_double[0][0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_double[0][1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_double[0][0], 1.6);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_complex_double[0][1], 2.2);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_integer_single[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_integer_single[0], 3);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_integer_double[0], 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.array_integer_double[0], 3);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_real_single[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_real_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_real_single[0].j, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_real_single[0].a, 1.5f);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].j, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].a[0], -1.5);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_complex_double[0].a[1], -2.1);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_integer_single[0].j, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_integer_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_integer_single[0].j, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_integer_single[0].a, 6);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_matrix, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_pattern[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_pattern[0].j, 2);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_pattern[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.matrix_coordinate_pattern[0].j, 3);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_real_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_real_single[0].a, 1.5f);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_real_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_real_single[0].a, 1.5f);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_double;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_complex_double[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_complex_double[0].a[0], 1.5);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_complex_double[0].a[1], 2.1);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_complex_double[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_complex_double[0].a[0], -1.5);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_complex_double[0].a[1], -2.1);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_integer_single[0].i, 3);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_integer_single[0].a, 5);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_integer_single[0].i, 2);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_integer_single[0].a, 6);
        }
        mtxdistfile2_free(&mtxdistfile2);
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
        struct mtxdistfile2 mtxdistfile2;
        enum mtxprecision precision = mtx_single;
        err = mtxdistfile2_fread_rowwise(
            &mtxdistfile2, precision, mtx_block_cyclic, 0, 1,
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
        TEST_ASSERT_EQ(mtxfile_vector, mtxdistfile2.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxdistfile2.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxdistfile2.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxdistfile2.header.symmetry);
        TEST_ASSERT_EQ(3, mtxdistfile2.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxdistfile2.size.num_columns);
        TEST_ASSERT_EQ(2, mtxdistfile2.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxdistfile2.precision);
        TEST_ASSERT_EQ(2, mtxdistfile2.datasize);
        if (rank == 0) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_pattern[0].i, 3);
        } else if (rank == 1) {
            TEST_ASSERT_EQ(1, mtxdistfile2.localdatasize);
            TEST_ASSERT_EQ(mtxdistfile2.data.vector_coordinate_pattern[0].i, 2);
        }
        mtxdistfile2_free(&mtxdistfile2);
        fclose(f);
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
    TEST_RUN(test_mtxdistfile2_from_mtxfile_rowwise);
    TEST_RUN(test_mtxdistfile2_fread_rowwise);
    TEST_SUITE_END();

    /* 3. Clean up and return. */
    MPI_Finalize();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
