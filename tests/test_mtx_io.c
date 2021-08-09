/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Unit tests for Matrix Market I/O.
 */

#include "test.h"

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/io.h>
#include <libmtx/matrix_coordinate.h>
#include <libmtx/mtx.h>
#include <libmtx/vector_array.h>
#include <libmtx/vector_coordinate.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>
#include <sys/wait.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtx_fread_header()` tests parsing Matrix Market headers.
 */
int test_mtx_fread_header(void)
{
    {
        /* Empty file. */
        int err;
        char mtxfile[] = "";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_HEADER, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /* Incomplete Matrix Market header. */
        int err;
        char mtxfile[] = "%MatrixMarket\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_HEADER, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /*
         * Test the maximum allowed line length, which occurs when a
         * newline character appears in position _SC_LINE_MAX-1.
         */
        int err;
        long int line_max = sysconf(_SC_LINE_MAX);
        char * mtxfile = malloc(line_max+1);
        for (int i = 0; i < line_max; i++)
            mtxfile[i] = 'a';
        mtxfile[line_max-1] = '\n';
        mtxfile[line_max] = '\0';
        FILE * f = fmemopen(mtxfile, strlen(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_NEQ_MSG(
            MTX_ERR_LINE_TOO_LONG, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
        free(mtxfile);
    }

    {
        /* Exceeding the maximum allowed line length should fail. */
        int err;
        long int line_max = sysconf(_SC_LINE_MAX);
        char * mtxfile = malloc(line_max+2);
        for (int i = 0; i < line_max+1; i++)
            mtxfile[i] = 'a';
        mtxfile[line_max] = '\n';
        mtxfile[line_max+1] = '\0';
        FILE * f = fmemopen(mtxfile, strlen(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_LINE_TOO_LONG, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
        free(mtxfile);
    }

    {
        /* Invalid object. */
        int err;
        char mtxfile[] = "%%MatrixMarket invalid_object";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_OBJECT, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /* Invalid format. */
        int err;
        char mtxfile[] = "%%MatrixMarket matrix invalid_format";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_FORMAT, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /* Invalid field. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate invalid_field";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_FIELD, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /* Invalid symmetry. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real invalid_symmetry";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_SYMMETRY, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /* Valid Matrix Market header. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real general\n"
            "0 0 0";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        TEST_ASSERT_EQ(mtx_matrix, mtx.object);
        TEST_ASSERT_EQ(mtx_coordinate, mtx.format);
        TEST_ASSERT_EQ(mtx_real, mtx.field);
        TEST_ASSERT_EQ(mtx_general, mtx.symmetry);
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_comment_lines()` tests parsing Matrix Market
 * comment lines.
 */
int test_mtx_fread_comment_lines(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix coordinate real general\n"
        "% First comment line\n"
        "% Second comment line\n"
        "0 0 0";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(2, mtx.num_comment_lines);
    TEST_ASSERT_STREQ("% First comment line\n", mtx.comment_lines[0]);
    TEST_ASSERT_STREQ("% Second comment line\n", mtx.comment_lines[1]);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_size_line()` tests parsing the size line
 * in Matrix Market files.
 */
int test_mtx_fread_matrix_size_line(void)
{
    {
        /* Dense matrix (array). */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix array real general\n"
            "0 0\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        TEST_ASSERT_EQ(0, mtx.num_rows);
        TEST_ASSERT_EQ(0, mtx.num_columns);
        TEST_ASSERT_EQ(0, mtx.num_nonzeros);
        TEST_ASSERT_EQ(0, mtx.size);
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        /* Sparse matrix (coordinate). */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real general\n"
            "0 0 0\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        TEST_ASSERT_EQ(0, mtx.num_rows);
        TEST_ASSERT_EQ(0, mtx.num_columns);
        TEST_ASSERT_EQ(0, mtx.num_nonzeros);
        TEST_ASSERT_EQ(0, mtx.size);
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] =
            "%%MatrixMarket vector array real general\n"
            "0\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        TEST_ASSERT_EQ(0, mtx.num_rows);
        TEST_ASSERT_EQ(-1, mtx.num_columns);
        TEST_ASSERT_EQ(0, mtx.num_nonzeros);
        TEST_ASSERT_EQ(0, mtx.size);
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] =
            "%%MatrixMarket vector coordinate real general\n"
            "0 0\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtx mtx;
        int line_number, column_number;
        err = mtx_fread(&mtx, f, &line_number, &column_number);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err,
            "%d:%d: %s", line_number, column_number,
            mtx_strerror(err));
        TEST_ASSERT_EQ(0, mtx.num_rows);
        TEST_ASSERT_EQ(-1, mtx.num_columns);
        TEST_ASSERT_EQ(0, mtx.num_nonzeros);
        TEST_ASSERT_EQ(0, mtx.size);
        if (!err)
            mtx_free(&mtx);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_array_real()` tests parsing data for a
 * dense matrix with real coefficients.
 */
int test_mtx_fread_matrix_array_real(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix array real general\n"
        "2 2\n"
        "1.0\n2.0\n3.0\n4.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(float), mtx.nonzero_size);
    const float * data = (const float *) mtx.data;
    TEST_ASSERT_EQ(1.0, data[0]);
    TEST_ASSERT_EQ(2.0, data[1]);
    TEST_ASSERT_EQ(3.0, data[2]);
    TEST_ASSERT_EQ(4.0, data[3]);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_array_double()` tests parsing data for a
 * dense matrix with double-precision real coefficients.
 */
int test_mtx_fread_matrix_array_double(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix array double general\n"
        "2 2\n"
        "1.0\n2.0\n3.0\n4.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(double), mtx.nonzero_size);
    const double * data = (const double *) mtx.data;
    TEST_ASSERT_EQ(1.0, data[0]);
    TEST_ASSERT_EQ(2.0, data[1]);
    TEST_ASSERT_EQ(3.0, data[2]);
    TEST_ASSERT_EQ(4.0, data[3]);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_array_complex()` tests parsing data for a
 * dense matrix with complex coefficients.
 */
int test_mtx_fread_matrix_array_complex(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix array complex general\n"
        "2 2\n"
        "1.0 2.0\n3.0 4.0\n5.0 6.0\n7.0 8.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(2*sizeof(float), mtx.nonzero_size);
    const float * data = (const float *) mtx.data;
    TEST_ASSERT_EQ(1.0, data[0]); TEST_ASSERT_EQ(2.0, data[1]);
    TEST_ASSERT_EQ(3.0, data[2]); TEST_ASSERT_EQ(4.0, data[3]);
    TEST_ASSERT_EQ(5.0, data[4]); TEST_ASSERT_EQ(6.0, data[5]);
    TEST_ASSERT_EQ(7.0, data[6]); TEST_ASSERT_EQ(8.0, data[7]);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_array_integer()` tests parsing data for a
 * dense matrix with integer coefficients.
 */
int test_mtx_fread_matrix_array_integer(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix array integer general\n"
        "2 2\n"
        "1\n2\n3\n4\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(2, mtx.num_rows);
    TEST_ASSERT_EQ(2, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(int), mtx.nonzero_size);
    const int * data = (const int *) mtx.data;
    TEST_ASSERT_EQ(1, data[0]);
    TEST_ASSERT_EQ(2, data[1]);
    TEST_ASSERT_EQ(3, data[2]);
    TEST_ASSERT_EQ(4, data[3]);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_coordinate_real()` tests parsing data for
 * a sparse matrix with real coefficients.
 */
int test_mtx_fread_matrix_coordinate_real(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "1 3 2.0\n"
        "2 2 3.0\n"
        "3 3 4.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_real * data =
        (const struct mtx_matrix_coordinate_real *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1,data[0].j); TEST_ASSERT_EQ(1.0,data[0].a);
    TEST_ASSERT_EQ(1,data[1].i); TEST_ASSERT_EQ(3,data[1].j); TEST_ASSERT_EQ(2.0,data[1].a);
    TEST_ASSERT_EQ(2,data[2].i); TEST_ASSERT_EQ(2,data[2].j); TEST_ASSERT_EQ(3.0,data[2].a);
    TEST_ASSERT_EQ(3,data[3].i); TEST_ASSERT_EQ(3,data[3].j); TEST_ASSERT_EQ(4.0,data[3].a);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_coordinate_double()` tests parsing data
 * for a sparse matrix with double-precision real coefficients.
 */
int test_mtx_fread_matrix_coordinate_double(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix coordinate double general\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "1 3 2.0\n"
        "2 2 3.0\n"
        "3 3 4.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_double), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_double * data =
        (const struct mtx_matrix_coordinate_double *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1,data[0].j); TEST_ASSERT_EQ(1.0,data[0].a);
    TEST_ASSERT_EQ(1,data[1].i); TEST_ASSERT_EQ(3,data[1].j); TEST_ASSERT_EQ(2.0,data[1].a);
    TEST_ASSERT_EQ(2,data[2].i); TEST_ASSERT_EQ(2,data[2].j); TEST_ASSERT_EQ(3.0,data[2].a);
    TEST_ASSERT_EQ(3,data[3].i); TEST_ASSERT_EQ(3,data[3].j); TEST_ASSERT_EQ(4.0,data[3].a);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_coordinate_complex()` tests parsing data
 * for a sparse matrix with complex coefficients.
 */
int test_mtx_fread_matrix_coordinate_complex(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix coordinate complex general\n"
        "3 3 4\n"
        "1 1 1.0 2.0\n"
        "1 3 3.0 4.0\n"
        "2 2 5.0 6.0\n"
        "3 3 7.0 8.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_complex), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_complex * data =
        (const struct mtx_matrix_coordinate_complex *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1,data[0].j);
    TEST_ASSERT_EQ(1.0,data[0].a); TEST_ASSERT_EQ(2.0,data[0].b);
    TEST_ASSERT_EQ(1,data[1].i); TEST_ASSERT_EQ(3,data[1].j);
    TEST_ASSERT_EQ(3.0,data[1].a); TEST_ASSERT_EQ(4.0,data[1].b);
    TEST_ASSERT_EQ(2,data[2].i); TEST_ASSERT_EQ(2,data[2].j);
    TEST_ASSERT_EQ(5.0,data[2].a); TEST_ASSERT_EQ(6.0,data[2].b);
    TEST_ASSERT_EQ(3,data[3].i); TEST_ASSERT_EQ(3,data[3].j);
    TEST_ASSERT_EQ(7.0,data[3].a); TEST_ASSERT_EQ(8.0,data[3].b);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_coordinate_integer()` tests parsing data
 * for a sparse matrix with integer coefficients.
 */
int test_mtx_fread_matrix_coordinate_integer(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 3 4\n"
        "1 1 1\n"
        "1 3 2\n"
        "2 2 3\n"
        "3 3 4\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_integer), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_integer * data =
        (const struct mtx_matrix_coordinate_integer *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1,data[0].j); TEST_ASSERT_EQ(1,data[0].a);
    TEST_ASSERT_EQ(1,data[1].i); TEST_ASSERT_EQ(3,data[1].j); TEST_ASSERT_EQ(2,data[1].a);
    TEST_ASSERT_EQ(2,data[2].i); TEST_ASSERT_EQ(2,data[2].j); TEST_ASSERT_EQ(3,data[2].a);
    TEST_ASSERT_EQ(3,data[3].i); TEST_ASSERT_EQ(3,data[3].j); TEST_ASSERT_EQ(4,data[3].a);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_matrix_coordinate_pattern()` tests parsing data
 * for a sparse matrix with boolean coefficients (pattern).
 */
int test_mtx_fread_matrix_coordinate_pattern(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket matrix coordinate pattern general\n"
        "3 3 4\n"
        "1 1\n"
        "1 3\n"
        "2 2\n"
        "3 3\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_pattern), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_pattern * data =
        (const struct mtx_matrix_coordinate_pattern *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1,data[0].j);
    TEST_ASSERT_EQ(1,data[1].i); TEST_ASSERT_EQ(3,data[1].j);
    TEST_ASSERT_EQ(2,data[2].i); TEST_ASSERT_EQ(2,data[2].j);
    TEST_ASSERT_EQ(3,data[3].i); TEST_ASSERT_EQ(3,data[3].j);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_vector_array_real()` tests parsing data for a
 * dense vector with real coefficients.
 */
int test_mtx_fread_vector_array_real(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket vector array real general\n"
        "4\n"
        "1.0\n2.0\n3.0\n4.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_vector, mtx.object);
    TEST_ASSERT_EQ(mtx_array, mtx.format);
    TEST_ASSERT_EQ(mtx_real, mtx.field);
    TEST_ASSERT_EQ(mtx_row_major, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_assembled, mtx.assembly);
    TEST_ASSERT_EQ(4, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(float), mtx.nonzero_size);
    const float * data = (const float *) mtx.data;
    TEST_ASSERT_EQ(1.0, data[0]);
    TEST_ASSERT_EQ(2.0, data[1]);
    TEST_ASSERT_EQ(3.0, data[2]);
    TEST_ASSERT_EQ(4.0, data[3]);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fread_vector_coordinate_real()` tests parsing data for
 * a sparse vector with real coefficients.
 */
int test_mtx_fread_vector_coordinate_real(void)
{
    int err;
    char mtxfile[] =
        "%%MatrixMarket vector coordinate real general\n"
        "8 4\n"
        "1 1.0\n"
        "3 2.0\n"
        "5 3.0\n"
        "6 4.0\n";
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
    TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_fread(&mtx, f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(8, mtx.num_rows);
    TEST_ASSERT_EQ(-1, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_vector_coordinate_real), mtx.nonzero_size);
    const struct mtx_vector_coordinate_real * data =
        (const struct mtx_vector_coordinate_real *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1.0,data[0].a);
    TEST_ASSERT_EQ(3,data[1].i); TEST_ASSERT_EQ(2.0,data[1].a);
    TEST_ASSERT_EQ(5,data[2].i); TEST_ASSERT_EQ(3.0,data[2].a);
    TEST_ASSERT_EQ(6,data[3].i); TEST_ASSERT_EQ(4.0,data[3].a);
    if (!err)
        mtx_free(&mtx);
    fclose(f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fwrite_matrix_coordinate_real()` tests writing a sparse
 * matrix with real coefficients to a file.
 */
int test_mtx_fwrite_matrix_coordinate_real(void)
{
    int err;

    /* Create a sparse matrix. */
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 3;
    int num_columns = 3;
    int64_t size = 4;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,1,1.0f},
        {1,3,2.0f},
        {2,2,3.0f},
        {3,3,4.0f}};
    err = mtx_init_matrix_coordinate_real(
        &mtx, mtx_general, mtx_nontriangular, mtx_unsorted,
        mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Write the matrix to file and verify the contents. */
    char mtxfile[1024] = {};
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "w");
    err = mtx_fwrite(&mtx, f, "%.1f");
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    fclose(f);
    mtx_free(&mtx);
    char expected_mtxfile[] =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "1 3 2.0\n"
        "2 2 3.0\n"
        "3 3 4.0\n";
    TEST_ASSERT_STREQ_MSG(expected_mtxfile, mtxfile, "\nexpected: %s\nactual: %s\n", expected_mtxfile, mtxfile);
    return TEST_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `test_mtx_gzread_matrix_coordinate_real()` tests reading a
 * compressed sparse matrix with real coefficients.
 */
int test_mtx_gzread_matrix_coordinate_real(void)
{
    int err;
    unsigned char test_mtx_gz[] = {
        0x1f, 0x8b, 0x08, 0x08, 0xb2, 0x36, 0xca, 0x60, 0x00, 0x03, 0x74, 0x65,
        0x73, 0x74, 0x32, 0x2e, 0x6d, 0x74, 0x78, 0x00, 0x53, 0x55, 0xf5, 0x4d,
        0x2c, 0x29, 0xca, 0xac, 0xf0, 0x4d, 0x2c, 0xca, 0x4e, 0x2d, 0x51, 0xc8,
        0x05, 0x73, 0x14, 0x92, 0xf3, 0xf3, 0x8b, 0x52, 0x32, 0xf3, 0x12, 0x4b,
        0x52, 0x15, 0x8a, 0x52, 0x13, 0x73, 0x14, 0xd2, 0x53, 0xf3, 0x52, 0x8b,
        0x12, 0x73, 0xb8, 0x8c, 0x15, 0x8c, 0x15, 0x4c, 0xb8, 0x0c, 0x15, 0x80,
        0x50, 0xcf, 0x00, 0x48, 0x1b, 0x2b, 0x18, 0x01, 0x69, 0x23, 0x05, 0x23,
        0x05, 0x63, 0x20, 0x0d, 0x96, 0x05, 0xd2, 0x00, 0x8b, 0xe5, 0xb5, 0x08,
        0x54, 0x00, 0x00, 0x00
    };

    /*
     * We cannot use fmemopen together with gzdopen, since there is no
     * file descriptor associated with the file stream returned by
     * fmemopen.  Instead, we open a pipe to get a file descriptor of
     * a buffer that only exists in memory.  Then we fork the process
     * and let the child process write the contents to the pipe. The
     * parent process can then read the file contents from the file
     * descriptor corresponding to the read end of the pipe.
     */
    int p[2];
    err = pipe(p);
    TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(errno));
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        /* The child process writes the matrix to file. */
        ssize_t bytes_written = write(p[1], test_mtx_gz, sizeof(test_mtx_gz));
        close(p[1]);
        if (bytes_written != sizeof(test_mtx_gz))
            exit(EXIT_FAILURE);
        exit(EXIT_SUCCESS);
    }
    close(p[1]);

    /* Wait for the child process to finish. */
    if (waitpid(pid, NULL, 0) == -1)
        TEST_FAIL_MSG("%s", strerror(errno));

    gzFile gz_f = gzdopen(p[0], "r");
    TEST_ASSERT_NEQ_MSG(NULL, gz_f, "%s", strerror(errno));
    struct mtx mtx;
    int line_number, column_number;
    err = mtx_gzread(&mtx, gz_f, &line_number, &column_number);
    TEST_ASSERT_EQ_MSG(
        MTX_SUCCESS, err,
        "%d:%d: %s", line_number, column_number,
        mtx_strerror(err));
    TEST_ASSERT_EQ(mtx_unsorted, mtx.sorting);
    TEST_ASSERT_EQ(mtx_unordered, mtx.ordering);
    TEST_ASSERT_EQ(mtx_unassembled, mtx.assembly);
    TEST_ASSERT_EQ(3, mtx.num_rows);
    TEST_ASSERT_EQ(3, mtx.num_columns);
    TEST_ASSERT_EQ(4, mtx.num_nonzeros);
    TEST_ASSERT_EQ(4, mtx.size);
    TEST_ASSERT_EQ(sizeof(struct mtx_matrix_coordinate_real), mtx.nonzero_size);
    const struct mtx_matrix_coordinate_real * data =
        (const struct mtx_matrix_coordinate_real *) mtx.data;
    TEST_ASSERT_EQ(1,data[0].i); TEST_ASSERT_EQ(1,data[0].j); TEST_ASSERT_EQ(1.0,data[0].a);
    TEST_ASSERT_EQ(1,data[1].i); TEST_ASSERT_EQ(3,data[1].j); TEST_ASSERT_EQ(2.0,data[1].a);
    TEST_ASSERT_EQ(2,data[2].i); TEST_ASSERT_EQ(2,data[2].j); TEST_ASSERT_EQ(3.0,data[2].a);
    TEST_ASSERT_EQ(3,data[3].i); TEST_ASSERT_EQ(3,data[3].j); TEST_ASSERT_EQ(4.0,data[3].a);
    if (!err)
        mtx_free(&mtx);
    gzclose(gz_f);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_gzwrite_matrix_coordinate_real()` tests writing a sparse
 * matrix with real coefficients to a file.
 */
int test_mtx_gzwrite_matrix_coordinate_real(void)
{
    int err;

    /* Create a sparse matrix. */
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 3;
    int num_columns = 3;
    int64_t size = 4;
    const struct mtx_matrix_coordinate_real data[] = {
        {1,1,1.0f},
        {1,3,2.0f},
        {2,2,3.0f},
        {3,3,4.0f}};
    err = mtx_init_matrix_coordinate_real(
        &mtx, mtx_general, mtx_nontriangular, mtx_unsorted,
        mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, num_columns, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /*
     * Write the matrix to file and verify the contents.
     *
     * Here we open a pipe to get a file descriptor of a buffer that
     * only exists in memory.
     */
    int p[2];
    err = pipe(p);
    TEST_ASSERT_EQ_MSG(0, err, "%s", strerror(errno));
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        /* The child process writes the matrix to file. */
        gzFile gz_f = gzdopen(p[1], "w");
        if (gz_f == Z_NULL) {
            fprintf(stdout, "FAIL:%s:%s:%d: "
                    "Assertion failed: Z_NULL == gz_f (%s)\n",
                    __FUNCTION__, __FILE__, __LINE__,
                    strerror(errno));
            mtx_free(&mtx);
            exit(EXIT_FAILURE);
        }
        err = mtx_gzwrite(&mtx, gz_f, "%.1f");
        if (err) {
            fprintf(stdout, "FAIL:%s:%s:%d: "
                    "Assertion failed: MTX_SUCCESS != %d (%s)\n",
                    __FUNCTION__, __FILE__, __LINE__,
                    err, mtx_strerror(err));
            gzclose(gz_f);
            mtx_free(&mtx);
            exit(EXIT_FAILURE);
        }
        gzclose(gz_f);
        mtx_free(&mtx);
        exit(EXIT_SUCCESS);
    }

    /* Wait for the child process to finish. */
    if (waitpid(pid, NULL, 0) == -1)
        TEST_FAIL_MSG("%s", strerror(errno));
    close(p[1]);
    mtx_free(&mtx);

    unsigned char expected_mtx_gz_file[] = {
        0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x53, 0x55,
        0xF5, 0x4D, 0x2C, 0x29, 0xCA, 0xAC, 0xF0, 0x4D, 0x2C, 0xCA, 0x4E, 0x2D,
        0x51, 0xC8, 0x05, 0x73, 0x14, 0x92, 0xF3, 0xF3, 0x8B, 0x52, 0x32, 0xF3,
        0x12, 0x4B, 0x52, 0x15, 0x8A, 0x52, 0x13, 0x73, 0x14, 0xD2, 0x53, 0xF3,
        0x52, 0x8B, 0x12, 0x73, 0xB8, 0x8C, 0x15, 0x8C, 0x15, 0x4C, 0xB8, 0x0C,
        0x15, 0x80, 0x50, 0xCF, 0x00, 0x48, 0x1B, 0x2B, 0x18, 0x01, 0x69, 0x23,
        0x05, 0x23, 0x05, 0x63, 0x20, 0x0D, 0x96, 0x05, 0xD2, 0x00, 0x8B, 0xE5,
        0xB5, 0x08, 0x54, 0x00, 0x00, 0x00
    };

    /* The parent process reads the resulting compressed matrix from
     * the pipe. */
    unsigned char mtx_gz_file[1024] = {};
    ssize_t bytes_read = read(p[0], mtx_gz_file, sizeof(expected_mtx_gz_file));
    if (bytes_read == -1)
        TEST_FAIL_MSG("%s", strerror(errno));
    close(p[0]);

#if 0
    fprintf(stdout, "gzip-compressed .mtx file: {\n");
    for (int i = 0; i < bytes_read-1; i++) {
        fprintf(stdout, "0x%02hhX, ", mtx_gz_file[i]);
    }
    if (bytes_read > 0)
        fprintf(stdout, "0x%02hhX", mtx_gz_file[bytes_read-1]);
    fprintf(stdout, "}\n");
#endif

    for (int i = 0; i < sizeof(expected_mtx_gz_file) && i < bytes_read; i++) {
        if (expected_mtx_gz_file[i] != mtx_gz_file[i]) {
            fprintf(stdout, "FAIL:%s:%s:%d: "
                    "Assertion failed: "
                    "expected_mtx_gz_file[%d] (0x%02X) != "
                    "mtx_gz_file[%d] (0x%02X)\n",
                    __FUNCTION__, __FILE__, __LINE__,
                    i, expected_mtx_gz_file[i], i, mtx_gz_file[i]);
            return TEST_FAILURE;
        }
    }
    return TEST_SUCCESS;
}
#endif

/**
 * `test_mtx_fwrite_vector_array_real()` tests writing a sparse
 * vector with real coefficients to a file.
 */
int test_mtx_fwrite_vector_array_real(void)
{
    int err;

    /* Create a sparse vector. */
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int64_t size = 4;
    const float data[] = {1.0f,2.0f,3.0f,4.0f};
    err = mtx_init_vector_array_real(
        &mtx, num_comment_lines, comment_lines, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Write the vector to file and verify the contents. */
    char mtxfile[1024] = {};
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "w");
    err = mtx_fwrite(&mtx, f, "%.1f");
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    fclose(f);
    mtx_free(&mtx);
    char expected_mtxfile[] =
        "%%MatrixMarket vector array real general\n"
        "4\n"
        "1.0\n"
        "2.0\n"
        "3.0\n"
        "4.0\n";
    TEST_ASSERT_STREQ_MSG(expected_mtxfile, mtxfile, "\nexpected: %s\nactual: %s\n", expected_mtxfile, mtxfile);
    return TEST_SUCCESS;
}

/**
 * `test_mtx_fwrite_vector_coordinate_real()` tests writing a sparse
 * vector with real coefficients to a file.
 */
int test_mtx_fwrite_vector_coordinate_real(void)
{
    int err;

    /* Create a sparse vector. */
    struct mtx mtx;
    int num_comment_lines = 0;
    const char * comment_lines[] = {};
    int num_rows = 4;
    int64_t size = 3;
    const struct mtx_vector_coordinate_real data[] = {
        {1,1.0f},
        {2,2.0f},
        {4,4.0f}};
    err = mtx_init_vector_coordinate_real(
        &mtx, mtx_unsorted, mtx_unordered, mtx_unassembled,
        num_comment_lines, comment_lines,
        num_rows, size, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));

    /* Write the vector to file and verify the contents. */
    char mtxfile[1024] = {};
    FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "w");
    err = mtx_fwrite(&mtx, f, "%.1f");
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
    fclose(f);
    mtx_free(&mtx);
    char expected_mtxfile[] =
        "%%MatrixMarket vector coordinate real general\n"
        "4 3\n"
        "1 1.0\n"
        "2 2.0\n"
        "4 4.0\n";
    TEST_ASSERT_STREQ_MSG(expected_mtxfile, mtxfile, "\nexpected: %s\nactual: %s\n", expected_mtxfile, mtxfile);
    return TEST_SUCCESS;
}

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market I/O\n");
    TEST_RUN(test_mtx_fread_header);
    TEST_RUN(test_mtx_fread_comment_lines);
    TEST_RUN(test_mtx_fread_matrix_size_line);
    TEST_RUN(test_mtx_fread_matrix_array_real);
    TEST_RUN(test_mtx_fread_matrix_array_double);
    TEST_RUN(test_mtx_fread_matrix_array_complex);
    TEST_RUN(test_mtx_fread_matrix_array_integer);
    TEST_RUN(test_mtx_fread_matrix_coordinate_real);
    TEST_RUN(test_mtx_fread_matrix_coordinate_double);
    TEST_RUN(test_mtx_fread_matrix_coordinate_complex);
    TEST_RUN(test_mtx_fread_matrix_coordinate_integer);
    TEST_RUN(test_mtx_fread_matrix_coordinate_pattern);
    TEST_RUN(test_mtx_fread_vector_array_real);
    TEST_RUN(test_mtx_fread_vector_coordinate_real);
    TEST_RUN(test_mtx_fwrite_matrix_coordinate_real);
#ifdef LIBMTX_HAVE_LIBZ
    TEST_RUN(test_mtx_gzread_matrix_coordinate_real);
    TEST_RUN(test_mtx_gzwrite_matrix_coordinate_real);
#endif
    TEST_RUN(test_mtx_fwrite_vector_array_real);
    TEST_RUN(test_mtx_fwrite_vector_coordinate_real);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
