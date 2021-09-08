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
 * Last modified: 2021-09-01
 *
 * Unit tests for Matrix Market files.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/coordinate.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>

#include <errno.h>
#include <unistd.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `test_mtxfile_parse_header()` tests parsing Matrix Market headers.
 */
int test_mtxfile_parse_header(void)
{
    {
        struct mtxfile_header header;
        const char line[] = "";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_HEADER, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%MatrixMarket";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_HEADER, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%MatrixMarketasdf";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_HEADER, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%%MatrixMarket invalid_object";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_OBJECT, err);
        TEST_ASSERT_EQ(15, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%%MatrixMarket matrix invalid_format";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_FORMAT, err);
        TEST_ASSERT_EQ(22, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%%MatrixMarket matrix coordinate invalid_field";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_FIELD, err);
        TEST_ASSERT_EQ(33, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%%MatrixMarket matrix coordinate real invalid_symmetry";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_SYMMETRY, err);
        TEST_ASSERT_EQ(38, bytes_read);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%%MatrixMarket matrix coordinate real general";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(mtxfile_matrix, header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, header.format);
        TEST_ASSERT_EQ(mtxfile_real, header.field);
        TEST_ASSERT_EQ(mtxfile_general, header.symmetry);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfile_header header;
        const char line[] = "%%MatrixMarket matrix coordinate real general\n";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_header(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(mtxfile_matrix, header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, header.format);
        TEST_ASSERT_EQ(mtxfile_real, header.field);
        TEST_ASSERT_EQ(mtxfile_general, header.symmetry);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_parse_size()` tests parsing Matrix Market size lines.
 */
int test_mtxfile_parse_size(void)
{
    {
        struct mtxfile_size size;
        const char line[] = "";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_size(
            &size, &bytes_read, &endptr, line, mtxfile_matrix, mtxfile_array);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_SIZE, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfile_size size;
        const char line[] = "8 10\n";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_size(
            &size, &bytes_read, &endptr, line, mtxfile_matrix, mtxfile_array);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 8);
        TEST_ASSERT_EQ(size.num_columns, 10);
        TEST_ASSERT_EQ(size.num_nonzeros, -1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfile_size size;
        const char line[] = "8 10 20\n";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_size(
            &size, &bytes_read, &endptr, line, mtxfile_matrix, mtxfile_coordinate);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 8);
        TEST_ASSERT_EQ(size.num_columns, 10);
        TEST_ASSERT_EQ(size.num_nonzeros, 20);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfile_size size;
        const char line[] = "10\n";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_size(
            &size, &bytes_read, &endptr, line, mtxfile_vector, mtxfile_array);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 10);
        TEST_ASSERT_EQ(size.num_columns, -1);
        TEST_ASSERT_EQ(size.num_nonzeros, -1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfile_size size;
        const char line[] = "10 8\n";
        int bytes_read = 0;
        const char * endptr;
        int err = mtxfile_parse_size(
            &size, &bytes_read, &endptr, line, mtxfile_vector, mtxfile_coordinate);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 10);
        TEST_ASSERT_EQ(size.num_columns, -1);
        TEST_ASSERT_EQ(size.num_nonzeros, 8);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_parse_data()` tests parsing Matrix Market data lines.
 */
int test_mtxfile_parse_data(void)
{
    /*
     * Array formats
     */

    {
        const char line[] = "1.5\n";
        int bytes_read = 0;
        const char * endptr;
        float data;
        int err = mtxfile_parse_data_array_real_single(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "1.5\n";
        int bytes_read = 0;
        const char * endptr;
        double data;
        int err = mtxfile_parse_data_array_real_double(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "1.5 2.1\n";
        int bytes_read = 0;
        const char * endptr;
        float data[2];
        int err = mtxfile_parse_data_array_complex_single(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data[0], 1.5f);
        TEST_ASSERT_EQ(data[1], 2.1f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "1.5 2.1\n";
        int bytes_read = 0;
        const char * endptr;
        double data[2];
        int err = mtxfile_parse_data_array_complex_double(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data[0], 1.5);
        TEST_ASSERT_EQ(data[1], 2.1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "2\n";
        int bytes_read = 0;
        const char * endptr;
        int32_t data;
        int err = mtxfile_parse_data_array_integer_single(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 2);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "2\n";
        int bytes_read = 0;
        const char * endptr;
        int64_t data;
        int err = mtxfile_parse_data_array_integer_double(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 2);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    /*
     * Matrix coordinate formats
     */

    {
        const char line[] = "3 2 1.5\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_real_single data;
        int err = mtxfile_parse_data_matrix_coordinate_real_single(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 1.5f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 1.5\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_real_double data;
        int err = mtxfile_parse_data_matrix_coordinate_real_double(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 1.5 2.1\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_complex_single data;
        int err = mtxfile_parse_data_matrix_coordinate_complex_single(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a[0], 1.5f); TEST_ASSERT_EQ(data.a[1], 2.1f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 1.5 2.1\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_complex_double data;
        int err = mtxfile_parse_data_matrix_coordinate_complex_double(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a[0], 1.5); TEST_ASSERT_EQ(data.a[1], 2.1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 4\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_integer_single data;
        int err = mtxfile_parse_data_matrix_coordinate_integer_single(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 4\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_integer_double data;
        int err = mtxfile_parse_data_matrix_coordinate_integer_double(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_matrix_coordinate_pattern data;
        int err = mtxfile_parse_data_matrix_coordinate_pattern(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    /*
     * Vector coordinate formats
     */

    {
        const char line[] = "3 1.5\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_real_single data;
        int err = mtxfile_parse_data_vector_coordinate_real_single(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 1.5f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 1.5\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_real_double data;
        int err = mtxfile_parse_data_vector_coordinate_real_double(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 1.5 2.1\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_complex_single data;
        int err = mtxfile_parse_data_vector_coordinate_complex_single(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a[0], 1.5f); TEST_ASSERT_EQ(data.a[1], 2.1f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 1.5 2.1\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_complex_double data;
        int err = mtxfile_parse_data_vector_coordinate_complex_double(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a[0], 1.5); TEST_ASSERT_EQ(data.a[1], 2.1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 4\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_integer_single data;
        int err = mtxfile_parse_data_vector_coordinate_integer_single(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 4\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_integer_double data;
        int err = mtxfile_parse_data_vector_coordinate_integer_double(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3\n";
        int bytes_read = 0;
        const char * endptr;
        struct mtxfile_vector_coordinate_pattern data;
        int err = mtxfile_parse_data_vector_coordinate_pattern(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_fread_header()` tests reading Matrix Market headers.
 */
int test_mtxfile_fread_header(void)
{
    {
        /* Empty file. */
        int err;
        char mtxfile[] = "";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_HEADER, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
        fclose(f);
    }

    {
        /* Incomplete Matrix Market header. */
        int err;
        char mtxfile[] = "%MatrixMarket\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_HEADER, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
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
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_NEQ_MSG(
            MTX_ERR_LINE_TOO_LONG, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
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
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_LINE_TOO_LONG, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
        fclose(f);
        free(mtxfile);
    }

    {
        /* Invalid object. */
        int err;
        char mtxfile[] = "%%MatrixMarket invalid_object";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_OBJECT, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
        fclose(f);
    }

    {
        /* Invalid format. */
        int err;
        char mtxfile[] = "%%MatrixMarket matrix invalid_format";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_FORMAT, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
        fclose(f);
    }

    {
        /* Invalid field. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate invalid_field";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_FIELD, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
        fclose(f);
    }

    {
        /* Invalid symmetry. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real invalid_symmetry";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_SYMMETRY, err, "%d: %s", lines_read+1,
            mtx_strerror(err));
        fclose(f);
    }

    {
        /* Valid Matrix Market header. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real general\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_header header;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_header(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%d: %s", lines_read+1, mtx_strerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, header.format);
        TEST_ASSERT_EQ(mtxfile_real, header.field);
        TEST_ASSERT_EQ(mtxfile_general, header.symmetry);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_fread_comments()` tests reading Matrix Market comment
 * lines from a stream.
 */
int test_mtxfile_fread_comments(void)
{
    {
        int err;
        char mtxfile[] = "Does not begin with '%'";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_comments comments;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_comments(&comments, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtx_strerror(err));
        TEST_ASSERT_EQ(0, lines_read);
        TEST_ASSERT_EQ(0, bytes_read);
        TEST_ASSERT_EQ(NULL, comments.root);
        mtxfile_comments_free(&comments);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] = "% Does not end with '\\n'";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_comments comments;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_comments(&comments, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_COMMENT, err);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] =
            "% First comment line\n"
            "% Second comment line\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_comments comments;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_comments(&comments, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%d: %s", lines_read+1, mtx_strerror(err));
        TEST_ASSERT_EQ(2, lines_read);
        TEST_ASSERT_EQ(strlen(mtxfile), bytes_read);
        TEST_ASSERT_NEQ(NULL, comments.root);
        TEST_ASSERT_STREQ("% First comment line\n", comments.root->comment_line);
        TEST_ASSERT_NEQ(NULL, comments.root->next);
        TEST_ASSERT_STREQ(
            "% Second comment line\n", comments.root->next->comment_line);
        mtxfile_comments_free(&comments);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_fread_size()` tests parsing the size line in Matrix
 * Market files.
 */
int test_mtxfile_fread_size(void)
{
    {
        int err;
        char mtxfile[] = "8 10\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_size size;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_size(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_matrix, mtxfile_array);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%d: %s", lines_read, mtx_strerror(err));
        TEST_ASSERT_EQ( 8, size.num_rows);
        TEST_ASSERT_EQ(10, size.num_columns);
        TEST_ASSERT_EQ(-1, size.num_nonzeros);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] = "10 8 16\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_size size;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_size(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_matrix, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%d: %s", lines_read, mtx_strerror(err));
        TEST_ASSERT_EQ(10, size.num_rows);
        TEST_ASSERT_EQ( 8, size.num_columns);
        TEST_ASSERT_EQ(16, size.num_nonzeros);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] = "4\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_size size;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_size(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_vector, mtxfile_array);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%d: %s", lines_read, mtx_strerror(err));
        TEST_ASSERT_EQ( 4, size.num_rows);
        TEST_ASSERT_EQ(-1, size.num_columns);
        TEST_ASSERT_EQ(-1, size.num_nonzeros);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] = "15 4\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfile_size size;
        int lines_read = 0;
        int bytes_read = 0;
        err = mtxfile_fread_size(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_vector, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%d: %s", lines_read, mtx_strerror(err));
        TEST_ASSERT_EQ(15, size.num_rows);
        TEST_ASSERT_EQ(-1, size.num_columns);
        TEST_ASSERT_EQ( 4, size.num_nonzeros);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_fread_data()` tests reading Matrix Market data lines
 * from a stream.
 */
int test_mtxfile_fread_data(void)
{
    /*
     * Array formats
     */

    {
        int err;
        char s[] = "1.5\n1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_array;
        enum mtxfile_field field = mtxfile_real;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_real_single[0], 1.5f);
        TEST_ASSERT_EQ(data.array_real_single[1], 1.6f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "1.5\n1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_array;
        enum mtxfile_field field = mtxfile_real;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_real_double[0], 1.5);
        TEST_ASSERT_EQ(data.array_real_double[1], 1.6);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "1.5 2.1\n1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_array;
        enum mtxfile_field field = mtxfile_complex;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_complex_single[0][0], 1.5f);
        TEST_ASSERT_EQ(data.array_complex_single[0][1], 2.1f);
        TEST_ASSERT_EQ(data.array_complex_single[1][0], 1.6f);
        TEST_ASSERT_EQ(data.array_complex_single[1][1], 2.2f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "1.5 2.1\n1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_array;
        enum mtxfile_field field = mtxfile_complex;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_complex_double[0][0], 1.5);
        TEST_ASSERT_EQ(data.array_complex_double[0][1], 2.1);
        TEST_ASSERT_EQ(data.array_complex_double[1][0], 1.6);
        TEST_ASSERT_EQ(data.array_complex_double[1][1], 2.2);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "2\n3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_array;
        enum mtxfile_field field = mtxfile_integer;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_integer_single[0], 2);
        TEST_ASSERT_EQ(data.array_integer_single[1], 3);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "2\n3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_array;
        enum mtxfile_field field = mtxfile_integer;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_integer_double[0], 2);
        TEST_ASSERT_EQ(data.array_integer_double[1], 3);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int err;
        char s[] = "3 2 1.5\n2 3 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_real;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[0].a, 1.5f);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[1].a, 1.5f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 2 1.5\n2 3 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_real;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[0].a, 1.5f);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[1].a, 1.5f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 2 1.5 2.1\n2 3 -1.5 -2.1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_complex;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[0].a[0], 1.5f);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[0].a[1], 2.1f);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[1].a[0], -1.5f);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_single[1].a[1], -2.1f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 2 1.5 2.1\n2 3 -1.5 -2.1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_complex;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[0].a[0], 1.5);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[0].a[1], 2.1);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[1].a[0], -1.5);
        TEST_ASSERT_EQ(data.matrix_coordinate_complex_double[1].a[1], -2.1);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 2 4\n2 3 4\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_integer;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[0].a, 4);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[1].a, 4);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 2 4\n2 3 4\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_integer;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[0].a, 4);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[1].a, 4);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 2\n2 3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_matrix;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_pattern;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[1].j, 3);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    /*
     * Vector coordinate formats
     */

    {
        int err;
        char s[] = "3 1.5\n4 1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_real;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[0].a, 1.5f);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[1].a, 1.6f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 1.5\n4 1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_real;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[0].a, 1.5);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[1].a, 1.6);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 1.5 2.1\n4 1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_complex;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[0].a[0], 1.5f);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[0].a[1], 2.1f);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[1].a[0], 1.6f);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[1].a[1], 2.2f);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 1.5 2.1\n4 1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_complex;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[0].a[0], 1.5);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[0].a[1], 2.1);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[1].a[0], 1.6);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[1].a[1], 2.2);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 4\n4 1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_integer;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[0].a, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[1].a, 1);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3 4\n4 1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_integer;
        enum mtx_precision precision = mtx_double;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[0].a, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[1].a, 1);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    {
        int err;
        char s[] = "3\n4\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        union mtxfile_data data;
        enum mtxfile_object object = mtxfile_vector;
        enum mtxfile_format format = mtxfile_coordinate;
        enum mtxfile_field field = mtxfile_pattern;
        enum mtx_precision precision = mtx_single;
        size_t size = 2;
        err = mtxfile_data_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfile_fread_data(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_pattern[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_pattern[1].i, 4);
        mtxfile_data_free(&data, object, format, field, precision);
    }

    return TEST_SUCCESS;
}

/**
 * `test_mtxfile_fread()` tests reading Matrix Market files from a
 * stream.
 */
int test_mtxfile_fread(void)
{
    /*
     * Array formats
     */

    {
        int err;
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.array_real_single[0], 1.5f);
        TEST_ASSERT_EQ(mtxfile.data.array_real_single[1], 1.6f);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.array_real_double[0], 1.5);
        TEST_ASSERT_EQ(mtxfile.data.array_real_double[1], 1.6);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_single[0][0], 1.5f);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_single[0][1], 2.1f);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_single[1][0], 1.6f);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_single[1][1], 2.2f);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_double[0][0], 1.5);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_double[0][1], 2.1);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_double[1][0], 1.6);
        TEST_ASSERT_EQ(mtxfile.data.array_complex_double[1][1], 2.2);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.array_integer_single[0], 2);
        TEST_ASSERT_EQ(mtxfile.data.array_integer_single[1], 3);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(2, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.array_integer_double[0], 2);
        TEST_ASSERT_EQ(mtxfile.data.array_integer_double[1], 3);
        mtxfile_free(&mtxfile);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate real general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5\n2 3 1.5\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[0].j, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[0].a, 1.5f);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[1].j, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[1].a, 1.5f);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate complex general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5 2.1\n"
            "2 3 -1.5 -2.1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[0].j, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[0].a[0], 1.5);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[0].a[1], 2.1);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[1].j, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[1].a[0], -1.5);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_complex_double[1].a[1], -2.1);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate integer general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 5\n2 3 6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_integer_single[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_integer_single[0].j, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_integer_single[0].a, 5);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_integer_single[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_integer_single[1].j, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_integer_single[1].a, 6);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate pattern general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2\n2 3\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(3, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_pattern[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_pattern[0].j, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_pattern[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_pattern[1].j, 3);
        mtxfile_free(&mtxfile);
    }

    /*
     * Vector coordinate formats
     */

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate real general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5\n2 1.5\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_real_single[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_real_single[0].a, 1.5f);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_real_single[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_real_single[1].a, 1.5f);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate complex general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5 2.1\n"
            "2 -1.5 -2.1\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_complex_double[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_complex_double[0].a[0], 1.5);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_complex_double[0].a[1], 2.1);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_complex_double[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_complex_double[1].a[0], -1.5);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_complex_double[1].a[1], -2.1);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate integer general\n"
            "% comment\n"
            "3 2\n"
            "3 5\n"
            "2 6\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_integer_single[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_integer_single[0].a, 5);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_integer_single[1].i, 2);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_integer_single[1].a, 6);
        mtxfile_free(&mtxfile);
    }

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate pattern general\n"
            "% comment\n"
            "3 2\n"
            "3\n"
            "2\n";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int lines_read = 0;
        int bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtx_precision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, f, &lines_read, &bytes_read, 0, NULL, precision);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(5, lines_read);
        TEST_ASSERT_EQ(mtxfile_vector, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(-1, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(2, mtxfile.size.num_nonzeros);
        TEST_ASSERT_EQ(precision, mtxfile.precision);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_pattern[0].i, 3);
        TEST_ASSERT_EQ(mtxfile.data.vector_coordinate_pattern[1].i, 2);
        mtxfile_free(&mtxfile);
    }

    return TEST_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * `test_mtx_gzread()` tests reading a sparse matrix with single
 * precision, real coefficients from a gzip-compressed stream.
 */
int test_mtxfile_gzread(void)
{
    int err;

    /* This is a gzip-compressed stream of the following file:
     *
     * %%MatrixMarket matrix coordinate real general
     * 3 3 4
     * 1 1 1.0
     * 1 3 2.0
     * 2 2 3.0
     * 3 3 4.0
     */
    unsigned char test_mtx_gz[] = {
        0x1f, 0x8b, 0x08, 0x08, 0xb2, 0x36, 0xca, 0x60, 0x00, 0x03, 0x74, 0x65,
        0x73, 0x74, 0x32, 0x2e, 0x6d, 0x74, 0x78, 0x00, 0x53, 0x55, 0xf5, 0x4d,
        0x2c, 0x29, 0xca, 0xac, 0xf0, 0x4d, 0x2c, 0xca, 0x4e, 0x2d, 0x51, 0xc8,
        0x05, 0x73, 0x14, 0x92, 0xf3, 0xf3, 0x8b, 0x52, 0x32, 0xf3, 0x12, 0x4b,
        0x52, 0x15, 0x8a, 0x52, 0x13, 0x73, 0x14, 0xd2, 0x53, 0xf3, 0x52, 0x8b,
        0x12, 0x73, 0xb8, 0x8c, 0x15, 0x8c, 0x15, 0x4c, 0xb8, 0x0c, 0x15, 0x80,
        0x50, 0xcf, 0x00, 0x48, 0x1b, 0x2b, 0x18, 0x01, 0x69, 0x23, 0x05, 0x23,
        0x05, 0x63, 0x20, 0x0d, 0x96, 0x05, 0xd2, 0x00, 0x8b, 0xe5, 0xb5, 0x08,
        0x54, 0x00, 0x00, 0x00 };

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
    int lines_read = 0;
    int bytes_read = 0;
    struct mtxfile mtxfile;
    enum mtx_precision precision = mtx_single;
    err = mtxfile_gzread(
        &mtxfile, gz_f, &lines_read, &bytes_read, 0, NULL, precision);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%d: %s",lines_read+1,mtx_strerror(err));
    TEST_ASSERT_EQ(84, bytes_read);
    TEST_ASSERT_EQ(6, lines_read);
    TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
    TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
    TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
    TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
    TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
    TEST_ASSERT_EQ(3, mtxfile.size.num_columns);
    TEST_ASSERT_EQ(4, mtxfile.size.num_nonzeros);
    TEST_ASSERT_EQ(precision, mtxfile.precision);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[0].i, 1);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[0].j, 1);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[0].a, 1.0f);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[1].i, 1);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[1].j, 3);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[1].a, 2.0f);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[2].i, 2);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[2].j, 2);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[2].a, 3.0f);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[3].i, 3);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[3].j, 3);
    TEST_ASSERT_EQ(mtxfile.data.matrix_coordinate_real_single[3].a, 4.0f);
    mtxfile_free(&mtxfile);
    gzclose(gz_f);
    return TEST_SUCCESS;
}
#endif

/**
 * `main()' entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market files\n");
    TEST_RUN(test_mtxfile_parse_header);
    TEST_RUN(test_mtxfile_parse_size);
    TEST_RUN(test_mtxfile_parse_data);
    TEST_RUN(test_mtxfile_fread_header);
    TEST_RUN(test_mtxfile_fread_comments);
    TEST_RUN(test_mtxfile_fread_size);
    TEST_RUN(test_mtxfile_fread_data);
    TEST_RUN(test_mtxfile_fread);
#ifdef LIBMTX_HAVE_LIBZ
    TEST_RUN(test_mtxfile_gzread);
#endif
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
