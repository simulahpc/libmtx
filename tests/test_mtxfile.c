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
 * Unit tests for Matrix Market files.
 */

#include "test.h"

#include <libmtx/error.h>
#include <libmtx/vector/precision.h>
#include <libmtx/mtxfile/comments.h>
#include <libmtx/mtxfile/data.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/partition.h>

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘test_mtxfileheader_parse()’ tests parsing Matrix Market headers.
 */
int test_mtxfileheader_parse(void)
{
    {
        struct mtxfileheader header;
        const char line[] = "";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_HEADER, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%MatrixMarket";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_HEADER, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%MatrixMarketasdf";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_HEADER, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%%MatrixMarket invalid_object";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_OBJECT, err);
        TEST_ASSERT_EQ(15, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%%MatrixMarket matrix invalid_format";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_FORMAT, err);
        TEST_ASSERT_EQ(22, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%%MatrixMarket matrix coordinate invalid_field";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_FIELD, err);
        TEST_ASSERT_EQ(33, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%%MatrixMarket matrix coordinate real invalid_symmetry";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
            &header, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_SYMMETRY, err);
        TEST_ASSERT_EQ(38, bytes_read);
    }

    {
        struct mtxfileheader header;
        const char line[] = "%%MatrixMarket matrix coordinate real general";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
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
        struct mtxfileheader header;
        const char line[] = "%%MatrixMarket matrix coordinate real general\n";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfileheader_parse(
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
 * ‘test_mtxfilesize_parse()’ tests parsing Matrix Market size lines.
 */
int test_mtxfilesize_parse(void)
{
    {
        struct mtxfilesize size;
        const char line[] = "";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfilesize_parse(
            &size, &bytes_read, &endptr, line, mtxfile_matrix, mtxfile_array);
        TEST_ASSERT_EQ(MTX_ERR_INVALID_MTX_SIZE, err);
        TEST_ASSERT_EQ(0, bytes_read);
    }

    {
        struct mtxfilesize size;
        const char line[] = "8 10";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfilesize_parse(
            &size, &bytes_read, &endptr, line, mtxfile_matrix, mtxfile_array);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 8);
        TEST_ASSERT_EQ(size.num_columns, 10);
        TEST_ASSERT_EQ(size.num_nonzeros, -1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfilesize size;
        const char line[] = "8 10 20";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfilesize_parse(
            &size, &bytes_read, &endptr, line, mtxfile_matrix, mtxfile_coordinate);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 8);
        TEST_ASSERT_EQ(size.num_columns, 10);
        TEST_ASSERT_EQ(size.num_nonzeros, 20);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfilesize size;
        const char line[] = "10";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfilesize_parse(
            &size, &bytes_read, &endptr, line, mtxfile_vector, mtxfile_array);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(size.num_rows, 10);
        TEST_ASSERT_EQ(size.num_columns, -1);
        TEST_ASSERT_EQ(size.num_nonzeros, -1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        struct mtxfilesize size;
        const char line[] = "10 8";
        int64_t bytes_read = 0;
        char * endptr;
        int err = mtxfilesize_parse(
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
 * ‘test_mtxfiledata_parse()’ tests parsing Matrix Market data lines.
 */
int test_mtxfiledata_parse(void)
{
    /*
     * Array formats
     */

    {
        const char line[] = "1.5";
        int64_t bytes_read = 0;
        char * endptr;
        float data;
        int err = mtxfiledata_parse_array_real_single(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "1.5";
        int64_t bytes_read = 0;
        char * endptr;
        double data;
        int err = mtxfiledata_parse_array_real_double(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "1.5 2.1";
        int64_t bytes_read = 0;
        char * endptr;
        float data[2];
        int err = mtxfiledata_parse_array_complex_single(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data[0], 1.5f);
        TEST_ASSERT_EQ(data[1], 2.1f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "1.5 2.1";
        int64_t bytes_read = 0;
        char * endptr;
        double data[2];
        int err = mtxfiledata_parse_array_complex_double(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data[0], 1.5);
        TEST_ASSERT_EQ(data[1], 2.1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "2";
        int64_t bytes_read = 0;
        char * endptr;
        int32_t data;
        int err = mtxfiledata_parse_array_integer_single(
            &data, &bytes_read, &endptr, line);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data, 2);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "2";
        int64_t bytes_read = 0;
        char * endptr;
        int64_t data;
        int err = mtxfiledata_parse_array_integer_double(
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
        const char line[] = "3 2 1.5";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_real_single data;
        int err = mtxfiledata_parse_matrix_coordinate_real_single(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 1.5f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 1.5";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_real_double data;
        int err = mtxfiledata_parse_matrix_coordinate_real_double(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 1.5 2.1";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_complex_single data;
        int err = mtxfiledata_parse_matrix_coordinate_complex_single(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a[0], 1.5f); TEST_ASSERT_EQ(data.a[1], 2.1f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 1.5 2.1";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_complex_double data;
        int err = mtxfiledata_parse_matrix_coordinate_complex_double(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a[0], 1.5); TEST_ASSERT_EQ(data.a[1], 2.1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 4";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_integer_single data;
        int err = mtxfiledata_parse_matrix_coordinate_integer_single(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2 4";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_integer_double data;
        int err = mtxfiledata_parse_matrix_coordinate_integer_double(
            &data, &bytes_read, &endptr, line, 4, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.j, 2);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 2";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_matrix_coordinate_pattern data;
        int err = mtxfiledata_parse_matrix_coordinate_pattern(
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
        const char line[] = "3 1.5";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_real_single data;
        int err = mtxfiledata_parse_vector_coordinate_real_single(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 1.5f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 1.5";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_real_double data;
        int err = mtxfiledata_parse_vector_coordinate_real_double(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 1.5);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 1.5 2.1";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_complex_single data;
        int err = mtxfiledata_parse_vector_coordinate_complex_single(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a[0], 1.5f); TEST_ASSERT_EQ(data.a[1], 2.1f);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 1.5 2.1";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_complex_double data;
        int err = mtxfiledata_parse_vector_coordinate_complex_double(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a[0], 1.5); TEST_ASSERT_EQ(data.a[1], 2.1);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 4";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_integer_single data;
        int err = mtxfiledata_parse_vector_coordinate_integer_single(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3 4";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_integer_double data;
        int err = mtxfiledata_parse_vector_coordinate_integer_double(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(data.a, 4);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    {
        const char line[] = "3";
        int64_t bytes_read = 0;
        char * endptr;
        struct mtxfile_vector_coordinate_pattern data;
        int err = mtxfiledata_parse_vector_coordinate_pattern(
            &data, &bytes_read, &endptr, line, 4);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(data.i, 3);
        TEST_ASSERT_EQ(strlen(line), bytes_read);
        TEST_ASSERT_EQ(line + strlen(line), endptr);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfileheader_fread()’ tests reading Matrix Market headers.
 */
int test_mtxfileheader_fread(void)
{
    {
        /* Empty file. */
        int err;
        char mtxfile[] = "";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_HEADER, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
        fclose(f);
    }

    {
        /* Incomplete Matrix Market header. */
        int err;
        char mtxfile[] = "%MatrixMarket\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_HEADER, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
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
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_NEQ_MSG(
            MTX_ERR_LINE_TOO_LONG, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
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
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_LINE_TOO_LONG, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
        fclose(f);
        free(mtxfile);
    }

    {
        /* Invalid object. */
        int err;
        char mtxfile[] = "%%MatrixMarket invalid_object";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_OBJECT, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
        fclose(f);
    }

    {
        /* Invalid format. */
        int err;
        char mtxfile[] = "%%MatrixMarket matrix invalid_format";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_FORMAT, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
        fclose(f);
    }

    {
        /* Invalid field. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate invalid_field";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_FIELD, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
        fclose(f);
    }

    {
        /* Invalid symmetry. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real invalid_symmetry";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_ERR_INVALID_MTX_SYMMETRY, err, "%"PRId64": %s", lines_read+1,
            mtxstrerror(err));
        fclose(f);
    }

    {
        /* Valid Matrix Market header. */
        int err;
        char mtxfile[] =
            "%%MatrixMarket matrix coordinate real general\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfileheader header;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfileheader_fread(&header, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%"PRId64": %s", lines_read+1, mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, header.format);
        TEST_ASSERT_EQ(mtxfile_real, header.field);
        TEST_ASSERT_EQ(mtxfile_general, header.symmetry);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_fread_comments()’ tests reading Matrix Market comment
 * lines from a stream.
 */
int test_mtxfile_fread_comments(void)
{
    {
        int err;
        char mtxfile[] = "Does not begin with '%'";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfilecomments comments;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfile_fread_comments(&comments, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(0, lines_read);
        TEST_ASSERT_EQ(0, bytes_read);
        TEST_ASSERT_EQ(NULL, comments.root);
        mtxfilecomments_free(&comments);
        fclose(f);
    }

    {
        int err;
        char mtxfile[] = "% Does not end with '\\n'";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfilecomments comments;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
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
        struct mtxfilecomments comments;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfile_fread_comments(&comments, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%"PRId64": %s", lines_read+1, mtxstrerror(err));
        TEST_ASSERT_EQ(2, lines_read);
        TEST_ASSERT_EQ(strlen(mtxfile), bytes_read);
        TEST_ASSERT_NEQ(NULL, comments.root);
        TEST_ASSERT_STREQ("% First comment line\n", comments.root->comment_line);
        TEST_ASSERT_NEQ(NULL, comments.root->next);
        TEST_ASSERT_STREQ(
            "% Second comment line\n", comments.root->next->comment_line);
        mtxfilecomments_free(&comments);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfilesize_fread()’ tests parsing the size line in Matrix
 * Market files.
 */
int test_mtxfilesize_fread(void)
{
    {
        int err;
        char mtxfile[] = "8 10\n";
        FILE * f = fmemopen(mtxfile, sizeof(mtxfile), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        struct mtxfilesize size;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfilesize_fread(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_matrix, mtxfile_array);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%"PRId64": %s", lines_read+1, mtxstrerror(err));
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
        struct mtxfilesize size;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfilesize_fread(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_matrix, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%"PRId64": %s", lines_read+1, mtxstrerror(err));
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
        struct mtxfilesize size;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfilesize_fread(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_vector, mtxfile_array);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%"PRId64": %s", lines_read+1, mtxstrerror(err));
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
        struct mtxfilesize size;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfilesize_fread(
            &size, f, &lines_read, &bytes_read, 0, NULL,
            mtxfile_vector, mtxfile_coordinate);
        TEST_ASSERT_EQ_MSG(
            MTX_SUCCESS, err, "%"PRId64": %s", lines_read+1, mtxstrerror(err));
        TEST_ASSERT_EQ(15, size.num_rows);
        TEST_ASSERT_EQ(-1, size.num_columns);
        TEST_ASSERT_EQ( 4, size.num_nonzeros);
        fclose(f);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfiledata_fread()’ tests reading Matrix Market data lines
 * from a stream.
 */
int test_mtxfiledata_fread(void)
{
    /*
     * Array formats
     */

    {
        int err;
        char s[] = "1.5\n1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_array;
        enum mtxfilefield field = mtxfile_real;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_real_single[0], 1.5f);
        TEST_ASSERT_EQ(data.array_real_single[1], 1.6f);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "1.5\n1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_array;
        enum mtxfilefield field = mtxfile_real;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_real_double[0], 1.5);
        TEST_ASSERT_EQ(data.array_real_double[1], 1.6);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "1.5 2.1\n1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_array;
        enum mtxfilefield field = mtxfile_complex;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_complex_single[0][0], 1.5f);
        TEST_ASSERT_EQ(data.array_complex_single[0][1], 2.1f);
        TEST_ASSERT_EQ(data.array_complex_single[1][0], 1.6f);
        TEST_ASSERT_EQ(data.array_complex_single[1][1], 2.2f);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "1.5 2.1\n1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_array;
        enum mtxfilefield field = mtxfile_complex;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_complex_double[0][0], 1.5);
        TEST_ASSERT_EQ(data.array_complex_double[0][1], 2.1);
        TEST_ASSERT_EQ(data.array_complex_double[1][0], 1.6);
        TEST_ASSERT_EQ(data.array_complex_double[1][1], 2.2);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "2\n3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_array;
        enum mtxfilefield field = mtxfile_integer;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_integer_single[0], 2);
        TEST_ASSERT_EQ(data.array_integer_single[1], 3);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "2\n3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_array;
        enum mtxfilefield field = mtxfile_integer;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, -1, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.array_integer_double[0], 2);
        TEST_ASSERT_EQ(data.array_integer_double[1], 3);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int err;
        char s[] = "3 2 1.5\n2 3 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_real;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[0].a, 1.5f);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_single[1].a, 1.5f);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 2 1.5\n2 3 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_real;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[0].a, 1.5f);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_real_double[1].a, 1.5f);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 2 1.5 2.1\n2 3 -1.5 -2.1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_complex;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
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
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 2 1.5 2.1\n2 3 -1.5 -2.1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_complex;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
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
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 2 4\n2 3 4";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_integer;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[0].a, 4);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_single[1].a, 4);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 2 4\n2 3 4";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_integer;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[0].a, 4);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[1].j, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_integer_double[1].a, 4);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 2\n2 3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_matrix;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_pattern;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, 4, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[0].i, 3);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[0].j, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[1].i, 2);
        TEST_ASSERT_EQ(data.matrix_coordinate_pattern[1].j, 3);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    /*
     * Vector coordinate formats
     */

    {
        int err;
        char s[] = "3 1.5\n4 1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_real;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[0].a, 1.5f);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_real_single[1].a, 1.6f);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 1.5\n4 1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_real;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[0].a, 1.5);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_real_double[1].a, 1.6);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 1.5 2.1\n4 1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_complex;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[0].a[0], 1.5f);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[0].a[1], 2.1f);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[1].a[0], 1.6f);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_single[1].a[1], 2.2f);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 1.5 2.1\n4 1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_complex;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[0].a[0], 1.5);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[0].a[1], 2.1);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[1].a[0], 1.6);
        TEST_ASSERT_EQ(data.vector_coordinate_complex_double[1].a[1], 2.2);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 4\n4 1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_integer;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[0].a, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_single[1].a, 1);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3 4\n4 1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_integer;
        enum mtxprecision precision = mtx_double;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[0].a, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[1].i, 4);
        TEST_ASSERT_EQ(data.vector_coordinate_integer_double[1].a, 1);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    {
        int err;
        char s[] = "3\n4";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        union mtxfiledata data;
        enum mtxfileobject object = mtxfile_vector;
        enum mtxfileformat format = mtxfile_coordinate;
        enum mtxfilefield field = mtxfile_pattern;
        enum mtxprecision precision = mtx_single;
        size_t size = 2;
        err = mtxfiledata_alloc(&data, object, format, field, precision, size);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        err = mtxfiledata_fread(
            &data, f, &lines_read, &bytes_read, 0, NULL,
            object, format, field, precision, 4, -1, size, 0);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ(strlen(s), bytes_read);
        TEST_ASSERT_EQ(size, lines_read);
        TEST_ASSERT_EQ(data.vector_coordinate_pattern[0].i, 3);
        TEST_ASSERT_EQ(data.vector_coordinate_pattern[1].i, 4);
        mtxfiledata_free(&data, object, format, field, precision);
        fclose(f);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_fread()’ tests reading Matrix Market files from a
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
            "1.5\n1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ(MTX_SUCCESS, err);
        TEST_ASSERT_EQ_MSG(strlen(s), bytes_read, "strlen(s)=%"PRId64" bytes_read=%"PRId64"", strlen(s), bytes_read);
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array real general\n"
            "% comment\n"
            "2 1\n"
            "1.5\n1.6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array complex general\n"
            "% comment\n"
            "2 1\n"
            "1.5 2.1\n1.6 2.2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix array integer general\n"
            "% comment\n"
            "2 1\n"
            "2\n3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
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
        fclose(f);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate real general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5\n2 3 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate complex general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 1.5 2.1\n"
            "2 3 -1.5 -2.1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate integer general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2 5\n2 3 6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket matrix coordinate pattern general\n"
            "% comment\n"
            "3 3 2\n"
            "3 2\n2 3";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    /*
     * Vector coordinate formats
     */

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate real general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5\n2 1.5";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate complex general\n"
            "% comment\n"
            "3 2\n"
            "3 1.5 2.1\n"
            "2 -1.5 -2.1";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_double;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate integer general\n"
            "% comment\n"
            "3 2\n"
            "3 5\n"
            "2 6";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    {
        int err;
        char s[] = "%%MatrixMarket vector coordinate pattern general\n"
            "% comment\n"
            "3 2\n"
            "3\n"
            "2";
        FILE * f = fmemopen(s, sizeof(s), "r");
        TEST_ASSERT_NEQ_MSG(NULL, f, "%s", strerror(errno));
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        struct mtxfile mtxfile;
        enum mtxprecision precision = mtx_single;
        err = mtxfile_fread(
            &mtxfile, precision, f, &lines_read, &bytes_read, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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
        fclose(f);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_fwrite()’ tests writing Matrix Market files to a
 * stream.
 */
int test_mtxfile_fwrite(void)
{
    int err;

    /*
     * Array formats
     */

    {
        struct mtxfile mtxfile;
        int num_rows = 3;
        int num_columns = 3;
        const double data[] = {
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0};
        err = mtxfile_init_matrix_array_real_double(
            &mtxfile, mtxfile_general, num_rows, num_columns, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxfile_fwrite(&mtxfile, f, "%.1f", &bytes_written);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        fclose(f);
        mtxfile_free(&mtxfile);
        char expected[] =
            "%%MatrixMarket matrix array real general\n"
            "3 3\n"
            "1.0\n" "2.0\n" "3.0\n"
            "4.0\n" "5.0\n" "6.0\n"
            "7.0\n" "8.0\n" "9.0\n";
        TEST_ASSERT_STREQ_MSG(
            expected, buf, "\nexpected: %s\nactual: %s\n",
            expected, buf);
    }

    {
        struct mtxfile mtxfile;
        int num_rows = 4;
        const double data[] = {1.0,2.0,3.0,4.0};
        err = mtxfile_init_vector_array_real_double(&mtxfile, num_rows, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxfile_fwrite(&mtxfile, f, "%.1f", &bytes_written);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        fclose(f);
        mtxfile_free(&mtxfile);
        char expected[] =
            "%%MatrixMarket vector array real general\n"
            "4\n"
            "1.0\n" "2.0\n" "3.0\n" "4.0\n";
        TEST_ASSERT_STREQ_MSG(
            expected, buf, "\nexpected: %s\nactual: %s\n",
            expected, buf);
    }

    /*
     * Matrix coordinate formats
     */

    {
        struct mtxfile mtxfile;
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_single data[] = {
            {1,1,1.0f},
            {1,3,2.0f},
            {2,2,3.0f},
            {3,3,4.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, num_nonzeros, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxfile_fwrite(&mtxfile, f, "%.1f", &bytes_written);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        fclose(f);
        mtxfile_free(&mtxfile);
        char expected[] =
            "%%MatrixMarket matrix coordinate real general\n"
            "3 3 4\n"
            "1 1 1.0\n"
            "1 3 2.0\n"
            "2 2 3.0\n"
            "3 3 4.0\n";
        TEST_ASSERT_STREQ_MSG(
            expected, buf, "\nexpected: %s\nactual: %s\n",
            expected, buf);
    }

    /*
     * Vector coordinate formats
     */

    {
        struct mtxfile mtxfile;
        int num_rows = 4;
        const struct mtxfile_vector_coordinate_real_single data[] = {
            {1,1.0f},
            {2,2.0f},
            {4,4.0f}};
        int64_t num_nonzeros = sizeof(data) / sizeof(*data);
        err = mtxfile_init_vector_coordinate_real_single(
            &mtxfile, num_rows, num_nonzeros, data);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        char buf[1024] = {};
        FILE * f = fmemopen(buf, sizeof(buf), "w");
        int64_t bytes_written;
        err = mtxfile_fwrite(&mtxfile, f, "%.1f", &bytes_written);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        fclose(f);
        mtxfile_free(&mtxfile);
        char expected[] =
            "%%MatrixMarket vector coordinate real general\n"
            "4 3\n"
            "1 1.0\n"
            "2 2.0\n"
            "4 4.0\n";
        TEST_ASSERT_STREQ_MSG(
            expected, buf, "\nexpected: %s\nactual: %s\n",
            expected, buf);
    }
    return TEST_SUCCESS;
}

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘test_mtx_gzread()’ tests reading a sparse matrix with single
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
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    struct mtxfile mtxfile;
    enum mtxprecision precision = mtx_single;
    err = mtxfile_gzread(
        &mtxfile, precision, gz_f, &lines_read, &bytes_read, 0, NULL);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%"PRId64": %s",lines_read+1,mtxstrerror(err));
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

/**
 * ‘test_mtxfile_gzwrite()’ tests writing a Matrix Market files to a
 * gzip-compressed stream.
 */
int test_mtxfile_gzwrite(void)
{
    int err;

    struct mtxfile mtx;
    int num_rows = 3;
    int num_columns = 3;
    int64_t num_nonzeros = 4;
    const struct mtxfile_matrix_coordinate_real_single data[] = {
        {1,1,1.0f},
        {1,3,2.0f},
        {2,2,3.0f},
        {3,3,4.0f}};
    err = mtxfile_init_matrix_coordinate_real_single(
        &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, data);
    TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

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
            mtxfile_free(&mtx);
            exit(EXIT_FAILURE);
        }
        err = mtxfile_gzwrite(&mtx, gz_f, "%.1f", NULL);
        if (err) {
            fprintf(stdout, "FAIL:%s:%s:%d: "
                    "Assertion failed: MTX_SUCCESS != %d (%s)\n",
                    __FUNCTION__, __FILE__, __LINE__,
                    err, mtxstrerror(err));
            gzclose(gz_f);
            mtxfile_free(&mtx);
            exit(EXIT_FAILURE);
        }
        gzclose(gz_f);
        mtxfile_free(&mtx);
        exit(EXIT_SUCCESS);
    }

    /* Wait for the child process to finish. */
    if (waitpid(pid, NULL, 0) == -1)
        TEST_FAIL_MSG("%s", strerror(errno));
    close(p[1]);
    mtxfile_free(&mtx);

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
        // Skip the 9th byte, which is the OS field, and varies from
        // one operating system to another (e.g., Linux and Mac OS may
        // use different values).  See RFC 1952 - GZIP file format
        // specification version 4.3
        // (https://datatracker.ietf.org/doc/html/rfc1952.html).
        if (i == 9)
            continue;

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
 * ‘test_mtxfile_transpose()’ tests transposing Matrix Market files.
 */
int test_mtxfile_transpose(void)
{
    int err;

    /*
     * Array formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_transpose(&mtx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(4.0, data[1]);
        TEST_ASSERT_EQ(7.0, data[2]);
        TEST_ASSERT_EQ(2.0, data[3]);
        TEST_ASSERT_EQ(5.0, data[4]);
        TEST_ASSERT_EQ(8.0, data[5]);
        TEST_ASSERT_EQ(3.0, data[6]);
        TEST_ASSERT_EQ(6.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 2;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_transpose(&mtx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(2, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(4.0, data[1]);
        TEST_ASSERT_EQ(2.0, data[2]);
        TEST_ASSERT_EQ(5.0, data[3]);
        TEST_ASSERT_EQ(3.0, data[4]);
        TEST_ASSERT_EQ(6.0, data[5]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        const double srcdata[] = {1.0, 2.0, 3.0};
        struct mtxfile mtx;
        err = mtxfile_init_vector_array_real_double(
            &mtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_transpose(&mtx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        mtxfile_free(&mtx);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1,1.0f}, {1,4,2.0f},
            {2,2,3.0f},
            {3,3,4.0f},
            {4,1,5.0f}, {4,4,6.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_transpose(&mtx);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(   4, data[1].i); TEST_ASSERT_EQ(   1, data[1].j);
        TEST_ASSERT_EQ(2.0f, data[1].a);
        TEST_ASSERT_EQ(   2, data[2].i); TEST_ASSERT_EQ(   2, data[2].j);
        TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(   3, data[3].i); TEST_ASSERT_EQ(   3, data[3].j);
        TEST_ASSERT_EQ(4.0f, data[3].a);
        TEST_ASSERT_EQ(   1, data[4].i); TEST_ASSERT_EQ(   4, data[4].j);
        TEST_ASSERT_EQ(5.0f, data[4].a);
        TEST_ASSERT_EQ(   4, data[5].i); TEST_ASSERT_EQ(   4, data[5].j);
        TEST_ASSERT_EQ(6.0f, data[5].a);
        mtxfile_free(&mtx);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_sort()’ tests sorting Matrix Market files.
 */
int test_mtxfile_sort(void)
{
    int err;

    /*
     * Array formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_column_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(4.0, data[1]);
        TEST_ASSERT_EQ(7.0, data[2]);
        TEST_ASSERT_EQ(2.0, data[3]);
        TEST_ASSERT_EQ(5.0, data[4]);
        TEST_ASSERT_EQ(8.0, data[5]);
        TEST_ASSERT_EQ(3.0, data[6]);
        TEST_ASSERT_EQ(6.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_morton, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(4.0, data[2]);
        TEST_ASSERT_EQ(5.0, data[3]);
        TEST_ASSERT_EQ(3.0, data[4]);
        TEST_ASSERT_EQ(6.0, data[5]);
        TEST_ASSERT_EQ(7.0, data[6]);
        TEST_ASSERT_EQ(8.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 2;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_column_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(2, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(4.0, data[1]);
        TEST_ASSERT_EQ(2.0, data[2]);
        TEST_ASSERT_EQ(5.0, data[3]);
        TEST_ASSERT_EQ(3.0, data[4]);
        TEST_ASSERT_EQ(6.0, data[5]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        const double srcdata[] = {1.0, 2.0, 3.0};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_vector_array_real_double(
            &mtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_column_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        mtxfile_free(&mtx);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {3,3,4.0f},
            {1,4,2.0f},
            {4,1,5.0f},
            {1,1,1.0f},
            {2,2,3.0f},
            {4,4,6.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_row_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(   1, data[1].i); TEST_ASSERT_EQ(   4, data[1].j);
        TEST_ASSERT_EQ(2.0f, data[1].a);
        TEST_ASSERT_EQ(   2, data[2].i); TEST_ASSERT_EQ(   2, data[2].j);
        TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(   3, data[3].i); TEST_ASSERT_EQ(   3, data[3].j);
        TEST_ASSERT_EQ(4.0f, data[3].a);
        TEST_ASSERT_EQ(   4, data[4].i); TEST_ASSERT_EQ(   1, data[4].j);
        TEST_ASSERT_EQ(5.0f, data[4].a);
        TEST_ASSERT_EQ(   4, data[5].i); TEST_ASSERT_EQ(   4, data[5].j);
        TEST_ASSERT_EQ(6.0f, data[5].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,4.0},
            {1,4,2.0},
            {4,1,5.0},
            {1,1,1.0},
            {2,2,3.0},
            {4,4,6.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_row_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtx.data.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0, data[0].a);
        TEST_ASSERT_EQ(  1, data[1].i); TEST_ASSERT_EQ(   4, data[1].j);
        TEST_ASSERT_EQ(2.0, data[1].a);
        TEST_ASSERT_EQ(  2, data[2].i); TEST_ASSERT_EQ(   2, data[2].j);
        TEST_ASSERT_EQ(3.0, data[2].a);
        TEST_ASSERT_EQ(  3, data[3].i); TEST_ASSERT_EQ(   3, data[3].j);
        TEST_ASSERT_EQ(4.0, data[3].a);
        TEST_ASSERT_EQ(  4, data[4].i); TEST_ASSERT_EQ(   1, data[4].j);
        TEST_ASSERT_EQ(5.0, data[4].a);
        TEST_ASSERT_EQ(  4, data[5].i); TEST_ASSERT_EQ(   4, data[5].j);
        TEST_ASSERT_EQ(6.0, data[5].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,4.0},
            {1,4,2.0},
            {4,1,5.0},
            {1,1,1.0},
            {2,2,3.0},
            {4,4,6.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_column_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtx.data.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(  1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0, data[0].a);
        TEST_ASSERT_EQ(  4, data[1].i); TEST_ASSERT_EQ(   1, data[1].j);
        TEST_ASSERT_EQ(5.0, data[1].a);
        TEST_ASSERT_EQ(  2, data[2].i); TEST_ASSERT_EQ(   2, data[2].j);
        TEST_ASSERT_EQ(3.0, data[2].a);
        TEST_ASSERT_EQ(  3, data[3].i); TEST_ASSERT_EQ(   3, data[3].j);
        TEST_ASSERT_EQ(4.0, data[3].a);
        TEST_ASSERT_EQ(  1, data[4].i); TEST_ASSERT_EQ(   4, data[4].j);
        TEST_ASSERT_EQ(2.0, data[4].a);
        TEST_ASSERT_EQ(  4, data[5].i); TEST_ASSERT_EQ(   4, data[5].j);
        TEST_ASSERT_EQ(6.0, data[5].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_complex_single srcdata[] = {
            {3,3,{4.0f,-4.0f}},
            {1,4,{2.0f,-2.0f}},
            {4,1,{5.0f,-5.0f}},
            {1,1,{1.0f,-1.0f}},
            {2,2,{3.0f,-3.0f}},
            {4,4,{6.0f,-6.0f}}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_complex_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_row_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_complex, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_complex_single * data =
            mtx.data.matrix_coordinate_complex_single;
        TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ( 1.0, data[0].a[0]); TEST_ASSERT_EQ(-1.0, data[0].a[1]);
        TEST_ASSERT_EQ(   1, data[1].i); TEST_ASSERT_EQ(   4, data[1].j);
        TEST_ASSERT_EQ( 2.0, data[1].a[0]); TEST_ASSERT_EQ(-2.0, data[1].a[1]);
        TEST_ASSERT_EQ(   2, data[2].i); TEST_ASSERT_EQ(   2, data[2].j);
        TEST_ASSERT_EQ( 3.0, data[2].a[0]); TEST_ASSERT_EQ(-3.0, data[2].a[1]);
        TEST_ASSERT_EQ(   3, data[3].i); TEST_ASSERT_EQ(   3, data[3].j);
        TEST_ASSERT_EQ( 4.0, data[3].a[0]); TEST_ASSERT_EQ(-4.0, data[3].a[1]);
        TEST_ASSERT_EQ(   4, data[4].i); TEST_ASSERT_EQ(   1, data[4].j);
        TEST_ASSERT_EQ( 5.0, data[4].a[0]); TEST_ASSERT_EQ(-5.0, data[4].a[1]);
        TEST_ASSERT_EQ(   4, data[5].i); TEST_ASSERT_EQ(   4, data[5].j);
        TEST_ASSERT_EQ( 6.0, data[5].a[0]); TEST_ASSERT_EQ(-6.0, data[5].a[1]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_integer_single srcdata[] = {
            {3,3,4},
            {1,4,2},
            {4,1,5},
            {1,1,1},
            {2,2,3},
            {4,4,6}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_integer_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_column_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_integer_single * data =
            mtx.data.matrix_coordinate_integer_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1, data[0].j);
        TEST_ASSERT_EQ(1, data[0].a);
        TEST_ASSERT_EQ(4, data[1].i); TEST_ASSERT_EQ(1, data[1].j);
        TEST_ASSERT_EQ(5, data[1].a);
        TEST_ASSERT_EQ(2, data[2].i); TEST_ASSERT_EQ(2, data[2].j);
        TEST_ASSERT_EQ(3, data[2].a);
        TEST_ASSERT_EQ(3, data[3].i); TEST_ASSERT_EQ(3, data[3].j);
        TEST_ASSERT_EQ(4, data[3].a);
        TEST_ASSERT_EQ(1, data[4].i); TEST_ASSERT_EQ(4, data[4].j);
        TEST_ASSERT_EQ(2, data[4].a);
        TEST_ASSERT_EQ(4, data[5].i); TEST_ASSERT_EQ(4, data[5].j);
        TEST_ASSERT_EQ(6, data[5].a);
        mtxfile_free(&mtx);
    }

    /*
     * Vector coordinate formats.
     */

    {
        int num_rows = 4;
        const struct mtxfile_vector_coordinate_integer_single srcdata[] = {
            {3,4}, {1,2}, {4,5}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtx, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_row_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(3, mtx.size.num_nonzeros);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtx.data.vector_coordinate_integer_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(2, data[0].a);
        TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(4, data[1].a);
        TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(5, data[2].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        const struct mtxfile_vector_coordinate_integer_single srcdata[] = {
            {3,4}, {1,2}, {4,5}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtx, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_column_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(3, mtx.size.num_nonzeros);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtx.data.vector_coordinate_integer_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(2, data[0].a);
        TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(4, data[1].a);
        TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(5, data[2].a);
        mtxfile_free(&mtx);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_compact()’ tests compacting Matrix Market files.
 */
int test_mtxfile_compact(void)
{
    int err;

    /*
     * Array formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_compact(&mtx, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        TEST_ASSERT_EQ(4.0, data[3]);
        TEST_ASSERT_EQ(5.0, data[4]);
        TEST_ASSERT_EQ(6.0, data[5]);
        TEST_ASSERT_EQ(7.0, data[6]);
        TEST_ASSERT_EQ(8.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        const double srcdata[] = {1.0, 2.0, 3.0};
        struct mtxfile mtx;
        err = mtxfile_init_vector_array_real_double(
            &mtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_compact(&mtx, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        mtxfile_free(&mtx);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1, 2.0f}, {1,2,-2.0f}, {2,1,-2.0f}, {2,2, 2.0f},
            {2,2, 2.0f}, {2,3,-2.0f}, {3,2,-2.0f}, {3,3, 2.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_compact(&mtx, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(7, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ( 2.0f, data[0].a);
        TEST_ASSERT_EQ(    1, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ(-2.0f, data[1].a);
        TEST_ASSERT_EQ(    2, data[2].i); TEST_ASSERT_EQ(   1, data[2].j);
        TEST_ASSERT_EQ(-2.0f, data[2].a);
        TEST_ASSERT_EQ(    2, data[3].i); TEST_ASSERT_EQ(   2, data[3].j);
        TEST_ASSERT_EQ( 4.0f, data[3].a);
        TEST_ASSERT_EQ(    2, data[4].i); TEST_ASSERT_EQ(   3, data[4].j);
        TEST_ASSERT_EQ(-2.0f, data[4].a);
        TEST_ASSERT_EQ(    3, data[5].i); TEST_ASSERT_EQ(   2, data[5].j);
        TEST_ASSERT_EQ(-2.0f, data[5].a);
        TEST_ASSERT_EQ(    3, data[6].i); TEST_ASSERT_EQ(   3, data[6].j);
        TEST_ASSERT_EQ( 2.0f, data[6].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1, 1.0f}, {1,1, 1.0f}, {1,1, 1.0f}, {2,2, 1.0f},
            {2,2, 1.0f}, {2,2, 1.0f}, {2,2, 1.0f}, {2,2, 1.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_compact(&mtx, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(2, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ( 3.0f, data[0].a);
        TEST_ASSERT_EQ(    2, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ( 5.0f, data[1].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {1,1, 2.0}, {1,2,-2.0}, {2,1,-2.0}, {2,2, 2.0},
            {2,2, 2.0}, {2,3,-2.0}, {3,2,-2.0}, {3,3, 2.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_compact(&mtx, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(7, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_double * data =
            mtx.data.matrix_coordinate_real_double;
        TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ( 2.0, data[0].a);
        TEST_ASSERT_EQ(   1, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ(-2.0, data[1].a);
        TEST_ASSERT_EQ(   2, data[2].i); TEST_ASSERT_EQ(   1, data[2].j);
        TEST_ASSERT_EQ(-2.0, data[2].a);
        TEST_ASSERT_EQ(   2, data[3].i); TEST_ASSERT_EQ(   2, data[3].j);
        TEST_ASSERT_EQ( 4.0, data[3].a);
        TEST_ASSERT_EQ(   2, data[4].i); TEST_ASSERT_EQ(   3, data[4].j);
        TEST_ASSERT_EQ(-2.0, data[4].a);
        TEST_ASSERT_EQ(   3, data[5].i); TEST_ASSERT_EQ(   2, data[5].j);
        TEST_ASSERT_EQ(-2.0, data[5].a);
        TEST_ASSERT_EQ(   3, data[6].i); TEST_ASSERT_EQ(   3, data[6].j);
        TEST_ASSERT_EQ( 2.0, data[6].a);
        mtxfile_free(&mtx);
    }

    /*
     * Vector coordinate formats.
     */

    {
        int num_rows = 4;
        const struct mtxfile_vector_coordinate_integer_single srcdata[] = {
            {3,4}, {1,2}, {1,1}, {4,5}, {4,4}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtx, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_compact(&mtx, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(3, mtx.size.num_nonzeros);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtx.data.vector_coordinate_integer_single;
        TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(4, data[0].a);
        TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(3, data[1].a);
        TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(9, data[2].a);
        mtxfile_free(&mtx);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_assemble()’ tests assembleing Matrix Market files.
 */
int test_mtxfile_assemble(void)
{
    int err;

    /*
     * Array formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_assemble(&mtx, mtxfile_row_major, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        TEST_ASSERT_EQ(4.0, data[3]);
        TEST_ASSERT_EQ(5.0, data[4]);
        TEST_ASSERT_EQ(6.0, data[5]);
        TEST_ASSERT_EQ(7.0, data[6]);
        TEST_ASSERT_EQ(8.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        const double srcdata[] = {1.0, 2.0, 3.0};
        struct mtxfile mtx;
        err = mtxfile_init_vector_array_real_double(
            &mtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_assemble(&mtx, mtxfile_row_major, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(1.0, data[0]);
        TEST_ASSERT_EQ(2.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        mtxfile_free(&mtx);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {2,2, 2.0f}, {2,1,-2.0f}, {1,2,-2.0f}, {1,1, 2.0f},
            {2,2, 2.0f}, {2,3,-2.0f}, {3,2,-2.0f}, {3,3, 2.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_assemble(&mtx, mtxfile_row_major, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(7, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ( 2.0f, data[0].a);
        TEST_ASSERT_EQ(    1, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ(-2.0f, data[1].a);
        TEST_ASSERT_EQ(    2, data[2].i); TEST_ASSERT_EQ(   1, data[2].j);
        TEST_ASSERT_EQ(-2.0f, data[2].a);
        TEST_ASSERT_EQ(    2, data[3].i); TEST_ASSERT_EQ(   2, data[3].j);
        TEST_ASSERT_EQ( 4.0f, data[3].a);
        TEST_ASSERT_EQ(    2, data[4].i); TEST_ASSERT_EQ(   3, data[4].j);
        TEST_ASSERT_EQ(-2.0f, data[4].a);
        TEST_ASSERT_EQ(    3, data[5].i); TEST_ASSERT_EQ(   2, data[5].j);
        TEST_ASSERT_EQ(-2.0f, data[5].a);
        TEST_ASSERT_EQ(    3, data[6].i); TEST_ASSERT_EQ(   3, data[6].j);
        TEST_ASSERT_EQ( 2.0f, data[6].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {2,2, 1.0f}, {1,1, 1.0f}, {1,1, 1.0f}, {2,2, 1.0f},
            {2,2, 1.0f}, {2,2, 1.0f}, {2,2, 1.0f}, {1,1, 1.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_assemble(&mtx, mtxfile_row_major, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(2, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    1, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ( 3.0f, data[0].a);
        TEST_ASSERT_EQ(    2, data[1].i); TEST_ASSERT_EQ(   2, data[1].j);
        TEST_ASSERT_EQ( 5.0f, data[1].a);
        mtxfile_free(&mtx);
    }

    /*
     * Vector coordinate formats.
     */

    {
        int num_rows = 4;
        const struct mtxfile_vector_coordinate_integer_single srcdata[] = {
            {3,4}, {1,2}, {1,1}, {4,5}, {4,4}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtx, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_assemble(&mtx, mtxfile_row_major, 0, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(3, mtx.size.num_nonzeros);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtx.data.vector_coordinate_integer_single;
        TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(3, data[0].a);
        TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(4, data[1].a);
        TEST_ASSERT_EQ(4, data[2].i); TEST_ASSERT_EQ(9, data[2].a);
        mtxfile_free(&mtx);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_partition_nonzeros()’ tests partitioning Matrix
 * Market files by nonzeros.
 */
int test_mtxfile_partition_nonzeros(void)
{
    int err;

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t partsizes[2] = {3,6};
        int64_t dstpartsizes[2] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_nonzeros(
            &src, mtx_block, 2, partsizes, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, dstpartsizes[0]);
        TEST_ASSERT_EQ(6, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(1, dstparts[3]);
        TEST_ASSERT_EQ(1, dstparts[4]);
        TEST_ASSERT_EQ(1, dstparts[5]);
        TEST_ASSERT_EQ(1, dstparts[6]);
        TEST_ASSERT_EQ(1, dstparts[7]);
        TEST_ASSERT_EQ(1, dstparts[8]);
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t dstpartsizes[2] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_nonzeros(
            &src, mtx_cyclic, 2, NULL, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(5, dstpartsizes[0]);
        TEST_ASSERT_EQ(4, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(1, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(1, dstparts[3]);
        TEST_ASSERT_EQ(0, dstparts[4]);
        TEST_ASSERT_EQ(1, dstparts[5]);
        TEST_ASSERT_EQ(0, dstparts[6]);
        TEST_ASSERT_EQ(1, dstparts[7]);
        TEST_ASSERT_EQ(0, dstparts[8]);
        mtxfile_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,9.0},{3,2,8.0},{2,2,5.0},{1,1,1.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t partsizes[2] = {2,2};
        int64_t dstpartsizes[2] = {};
        int dstparts[4] = {};
        err = mtxfile_partition_nonzeros(
            &src, mtx_block, 2, partsizes, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, dstpartsizes[0]);
        TEST_ASSERT_EQ(2, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(1, dstparts[2]);
        TEST_ASSERT_EQ(1, dstparts[3]);
        mtxfile_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_partition_rowwise()’ tests partitioning Matrix Market
 * files by rows.
 */
int test_mtxfile_partition_rowwise(void)
{
    int err;

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t partsizes[2] = {2,1};
        int64_t dstpartsizes[2] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_rowwise(
            &src, mtx_block, 2, partsizes, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, dstpartsizes[0]);
        TEST_ASSERT_EQ(3, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        TEST_ASSERT_EQ(0, dstparts[4]);
        TEST_ASSERT_EQ(0, dstparts[5]);
        TEST_ASSERT_EQ(1, dstparts[6]);
        TEST_ASSERT_EQ(1, dstparts[7]);
        TEST_ASSERT_EQ(1, dstparts[8]);
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t dstpartsizes[2] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_rowwise(
            &src, mtx_cyclic, 2, NULL, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, dstpartsizes[0]);
        TEST_ASSERT_EQ(3, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(1, dstparts[3]);
        TEST_ASSERT_EQ(1, dstparts[4]);
        TEST_ASSERT_EQ(1, dstparts[5]);
        TEST_ASSERT_EQ(0, dstparts[6]);
        TEST_ASSERT_EQ(0, dstparts[7]);
        TEST_ASSERT_EQ(0, dstparts[8]);
        mtxfile_free(&src);
    }

    {
        int num_rows = 8;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        struct mtxfile src;
        err = mtxfile_init_vector_array_real_double(&src, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t dstpartsizes[3] = {};
        int dstparts[8] = {};
        err = mtxfile_partition_rowwise(
            &src, mtx_block, 3, NULL, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, dstpartsizes[0]);
        TEST_ASSERT_EQ(3, dstpartsizes[1]);
        TEST_ASSERT_EQ(2, dstpartsizes[2]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(1, dstparts[3]);
        TEST_ASSERT_EQ(1, dstparts[4]);
        TEST_ASSERT_EQ(1, dstparts[5]);
        TEST_ASSERT_EQ(2, dstparts[6]);
        TEST_ASSERT_EQ(2, dstparts[7]);
        mtxfile_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,9.0},{3,2,8.0},{2,2,5.0},{1,1,1.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t partsizes[2] = {2,1};
        int64_t dstpartsizes[2] = {};
        int dstparts[4] = {};
        err = mtxfile_partition_rowwise(
            &src, mtx_block, 2, partsizes, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, dstpartsizes[0]);
        TEST_ASSERT_EQ(2, dstpartsizes[1]);
        TEST_ASSERT_EQ(1, dstparts[0]);
        TEST_ASSERT_EQ(1, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        mtxfile_free(&src);
    }

    {
        int num_rows = 8;
        const struct mtxfile_vector_coordinate_real_double srcdata[] = {
            {1,1.0},{2,2.0},{4,4.0},{5,5.0},{8,8.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t dstpartsizes[3] = {};
        int dstparts[5] = {};
        err = mtxfile_partition_rowwise(
            &src, mtx_block, 3, NULL, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, dstpartsizes[0]);
        TEST_ASSERT_EQ(2, dstpartsizes[1]);
        TEST_ASSERT_EQ(1, dstpartsizes[2]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(1, dstparts[2]);
        TEST_ASSERT_EQ(1, dstparts[3]);
        TEST_ASSERT_EQ(2, dstparts[4]);
        mtxfile_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_partition_columnwise()’ tests partitioning Matrix Market
 * files by columns.
 */
int test_mtxfile_partition_columnwise(void)
{
    int err;

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t partsizes[2] = {2,1};
        int64_t dstpartsizes[2] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_columnwise(
            &src, mtx_block, 2, partsizes, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, dstpartsizes[0]);
        TEST_ASSERT_EQ(3, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(1, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        TEST_ASSERT_EQ(0, dstparts[4]);
        TEST_ASSERT_EQ(1, dstparts[5]);
        TEST_ASSERT_EQ(0, dstparts[6]);
        TEST_ASSERT_EQ(0, dstparts[7]);
        TEST_ASSERT_EQ(1, dstparts[8]);
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t dstpartsizes[2] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_columnwise(
            &src, mtx_cyclic, 2, NULL, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(6, dstpartsizes[0]);
        TEST_ASSERT_EQ(3, dstpartsizes[1]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(1, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        TEST_ASSERT_EQ(1, dstparts[4]);
        TEST_ASSERT_EQ(0, dstparts[5]);
        TEST_ASSERT_EQ(0, dstparts[6]);
        TEST_ASSERT_EQ(1, dstparts[7]);
        TEST_ASSERT_EQ(0, dstparts[8]);
        mtxfile_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,9.0},{3,2,8.0},{2,2,5.0},{1,1,1.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t partsizes[2] = {2,1};
        int64_t dstpartsizes[2] = {};
        int dstparts[4] = {};
        err = mtxfile_partition_columnwise(
            &src, mtx_block, 2, partsizes, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, dstpartsizes[0]);
        TEST_ASSERT_EQ(1, dstpartsizes[1]);
        TEST_ASSERT_EQ(1, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        mtxfile_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_partition_2d()’ tests partitioning Matrix Market
 * files by columns.
 */
int test_mtxfile_partition_2d(void)
{
    int err;

    /*
     * Array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t rowpartsizes[2] = {2,1};
        int64_t colpartsizes[2] = {2,1};
        int64_t dstpartsizes[4] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_2d(
            &src, mtx_block, 2, rowpartsizes, 0, NULL,
            mtx_block, 2, colpartsizes, 0, NULL,
            dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, dstpartsizes[0]);
        TEST_ASSERT_EQ(2, dstpartsizes[1]);
        TEST_ASSERT_EQ(2, dstpartsizes[2]);
        TEST_ASSERT_EQ(1, dstpartsizes[3]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(0, dstparts[1]);
        TEST_ASSERT_EQ(1, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        TEST_ASSERT_EQ(0, dstparts[4]);
        TEST_ASSERT_EQ(1, dstparts[5]);
        TEST_ASSERT_EQ(2, dstparts[6]);
        TEST_ASSERT_EQ(2, dstparts[7]);
        TEST_ASSERT_EQ(3, dstparts[8]);
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t dstpartsizes[4] = {};
        int dstparts[9] = {};
        err = mtxfile_partition_2d(
            &src, mtx_cyclic, 2, NULL, 0, NULL,
            mtx_cyclic, 2, NULL, 0, NULL, dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(4, dstpartsizes[0]);
        TEST_ASSERT_EQ(2, dstpartsizes[1]);
        TEST_ASSERT_EQ(2, dstpartsizes[2]);
        TEST_ASSERT_EQ(1, dstpartsizes[3]);
        TEST_ASSERT_EQ(0, dstparts[0]);
        TEST_ASSERT_EQ(1, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(2, dstparts[3]);
        TEST_ASSERT_EQ(3, dstparts[4]);
        TEST_ASSERT_EQ(2, dstparts[5]);
        TEST_ASSERT_EQ(0, dstparts[6]);
        TEST_ASSERT_EQ(1, dstparts[7]);
        TEST_ASSERT_EQ(0, dstparts[8]);
        mtxfile_free(&src);
    }

    /*
     * Coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,9.0},{3,2,8.0},{2,2,5.0},{1,1,1.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        int64_t rowpartsizes[2] = {2,1};
        int64_t colpartsizes[2] = {2,1};
        int64_t dstpartsizes[4] = {};
        int dstparts[4] = {};
        err = mtxfile_partition_2d(
            &src, mtx_block, 2, rowpartsizes, 0, NULL,
            mtx_block, 2, colpartsizes, 0, NULL,
            dstparts, dstpartsizes);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(2, dstpartsizes[0]);
        TEST_ASSERT_EQ(0, dstpartsizes[1]);
        TEST_ASSERT_EQ(1, dstpartsizes[2]);
        TEST_ASSERT_EQ(1, dstpartsizes[3]);
        TEST_ASSERT_EQ(3, dstparts[0]);
        TEST_ASSERT_EQ(2, dstparts[1]);
        TEST_ASSERT_EQ(0, dstparts[2]);
        TEST_ASSERT_EQ(0, dstparts[3]);
        mtxfile_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_split()’ tests splitting Matrix Market files into
 * several parts.
 */
int test_mtxfile_split(void)
{
    int err;

    /*
     * array formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 2;
        int parts[] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
        int64_t num_rows_per_part[] = {2, 1};
        int64_t num_columns_per_part[] = {3, 3};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1]};
        err = mtxfile_split(
            num_parts, dstptrs, &src, size, parts,
            num_rows_per_part, num_columns_per_part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            const double * data = dsts[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(4.0, data[3]);
            TEST_ASSERT_EQ(5.0, data[4]);
            TEST_ASSERT_EQ(6.0, data[5]);
            mtxfile_free(&dsts[0]);
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
            const double * data = dsts[1].data.array_real_double;
            TEST_ASSERT_EQ(7.0, data[0]);
            TEST_ASSERT_EQ(8.0, data[1]);
            TEST_ASSERT_EQ(9.0, data[2]);
            mtxfile_free(&dsts[1]);
        }
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 2;
        int parts[] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
        int64_t num_rows_per_part[] = {2, 1};
        int64_t num_columns_per_part[] = {3, 3};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1]};
        err = mtxfile_split(
            num_parts, dstptrs, &src, size, parts,
            num_rows_per_part, num_columns_per_part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            const double * data = dsts[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            TEST_ASSERT_EQ(7.0, data[3]);
            TEST_ASSERT_EQ(8.0, data[4]);
            TEST_ASSERT_EQ(9.0, data[5]);
            mtxfile_free(&dsts[0]);
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
            const double * data = dsts[1].data.array_real_double;
            TEST_ASSERT_EQ(4.0, data[0]);
            TEST_ASSERT_EQ(5.0, data[1]);
            TEST_ASSERT_EQ(6.0, data[2]);
            mtxfile_free(&dsts[1]);
        }
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 2;
        int parts[] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
        int64_t num_rows_per_part[] = {3, 3};
        int64_t num_columns_per_part[] = {2, 1};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1]};
        err = mtxfile_split(
            num_parts, dstptrs, &src, size, parts,
            num_rows_per_part, num_columns_per_part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(3, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            const double * data = dsts[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(4.0, data[2]);
            TEST_ASSERT_EQ(5.0, data[3]);
            TEST_ASSERT_EQ(7.0, data[4]);
            TEST_ASSERT_EQ(8.0, data[5]);
            mtxfile_free(&dsts[0]);
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
            const double * data = dsts[1].data.array_real_double;
            TEST_ASSERT_EQ(3.0, data[0]);
            TEST_ASSERT_EQ(6.0, data[1]);
            TEST_ASSERT_EQ(9.0, data[2]);
            mtxfile_free(&dsts[1]);
        }
        mtxfile_free(&src);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile src;
        err = mtxfile_init_matrix_array_real_double(
            &src, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 4;
        int parts[] = {0, 0, 1, 0, 0, 1, 2, 2, 3};
        int64_t num_rows_per_part[] = {2, 2, 1, 1};
        int64_t num_columns_per_part[] = {2, 1, 2, 1};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1], &dsts[2], &dsts[3]};
        err = mtxfile_split(
            num_parts, dstptrs, &src, size, parts,
            num_rows_per_part, num_columns_per_part);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(2, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(2, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            const double * data = dsts[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(4.0, data[2]);
            TEST_ASSERT_EQ(5.0, data[3]);
            mtxfile_free(&dsts[0]);
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
            const double * data = dsts[1].data.array_real_double;
            TEST_ASSERT_EQ(3.0, data[0]);
            TEST_ASSERT_EQ(6.0, data[1]);
            mtxfile_free(&dsts[1]);
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
            const double * data = dsts[2].data.array_real_double;
            TEST_ASSERT_EQ(7.0, data[0]);
            TEST_ASSERT_EQ(8.0, data[1]);
            mtxfile_free(&dsts[2]);
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
            const double * data = dsts[3].data.array_real_double;
            TEST_ASSERT_EQ(9.0, data[0]);
            mtxfile_free(&dsts[3]);
        }
        mtxfile_free(&src);
    }

    {
        int num_rows = 8;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        struct mtxfile src;
        err = mtxfile_init_vector_array_real_double(&src, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 3;
        int parts[] = {0, 0, 0, 1, 1, 1, 2, 2};
        int64_t num_rows_per_part[] = {3, 3, 2};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1], &dsts[2]};
        err = mtxfile_split(
            num_parts, dstptrs, &src, size, parts, num_rows_per_part, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_array, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(3, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_nonzeros);
            const double * data = dsts[0].data.array_real_double;
            TEST_ASSERT_EQ(1.0, data[0]);
            TEST_ASSERT_EQ(2.0, data[1]);
            TEST_ASSERT_EQ(3.0, data[2]);
            mtxfile_free(&dsts[0]);
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
            const double * data = dsts[1].data.array_real_double;
            TEST_ASSERT_EQ(4.0, data[0]);
            TEST_ASSERT_EQ(5.0, data[1]);
            TEST_ASSERT_EQ(6.0, data[2]);
            mtxfile_free(&dsts[1]);
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
            const double * data = dsts[2].data.array_real_double;
            TEST_ASSERT_EQ(7.0, data[0]);
            TEST_ASSERT_EQ(8.0, data[1]);
            mtxfile_free(&dsts[2]);
        }
        mtxfile_free(&src);
    }

    /*
     * Matrix coordinate formats
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const struct mtxfile_matrix_coordinate_real_double srcdata[] = {
            {3,3,9.0},{3,2,8.0},{2,2,5.0},{1,1,1.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_matrix_coordinate_real_double(
            &src, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 2;
        int parts[] = {1, 1, 0, 0};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1]};
        err = mtxfile_split(num_parts, dstptrs, &src, size, parts, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(3, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(2, dsts[0].size.num_nonzeros);
            const struct mtxfile_matrix_coordinate_real_double * data =
                dsts[0].data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(2, data[0].i); TEST_ASSERT_EQ(2, data[0].j);
            TEST_ASSERT_EQ(5.0, data[0].a);
            TEST_ASSERT_EQ(1, data[1].i); TEST_ASSERT_EQ(1, data[1].j);
            TEST_ASSERT_EQ(1.0, data[1].a);
            mtxfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_matrix, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(3, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(3, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(2, dsts[1].size.num_nonzeros);
            const struct mtxfile_matrix_coordinate_real_double * data =
                dsts[1].data.matrix_coordinate_real_double;
            TEST_ASSERT_EQ(3, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
            TEST_ASSERT_EQ(9.0, data[0].a);
            TEST_ASSERT_EQ(3, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
            TEST_ASSERT_EQ(8.0, data[1].a);
            mtxfile_free(&dsts[1]);
        }
        mtxfile_free(&src);
    }

    {
        int num_rows = 8;
        const struct mtxfile_vector_coordinate_real_double srcdata[] = {
            {1,1.0},{2,2.0},{4,4.0},{5,5.0},{8,8.0}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile src;
        err = mtxfile_init_vector_coordinate_real_double(
            &src, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        const int num_parts = 3;
        int parts[] = {0, 0, 1, 1, 2};
        int size = sizeof(parts) / sizeof(*parts);
        struct mtxfile dsts[num_parts];
        struct mtxfile * dstptrs[] = {&dsts[0], &dsts[1], &dsts[2]};
        err = mtxfile_split(num_parts, dstptrs, &src, size, parts, NULL, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[0].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[0].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[0].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[0].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[0].precision);
            TEST_ASSERT_EQ(8, dsts[0].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[0].size.num_columns);
            TEST_ASSERT_EQ(2, dsts[0].size.num_nonzeros);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[0].data.vector_coordinate_real_double;
            TEST_ASSERT_EQ(1, data[0].i); TEST_ASSERT_EQ(1.0, data[0].a);
            TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2.0, data[1].a);
            mtxfile_free(&dsts[0]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[1].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[1].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[1].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[1].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[1].precision);
            TEST_ASSERT_EQ(8, dsts[1].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[1].size.num_columns);
            TEST_ASSERT_EQ(2, dsts[1].size.num_nonzeros);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[1].data.vector_coordinate_real_double;
            TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(4.0, data[0].a);
            TEST_ASSERT_EQ(5, data[1].i); TEST_ASSERT_EQ(5.0, data[1].a);
            mtxfile_free(&dsts[1]);
        }
        {
            TEST_ASSERT_EQ(mtxfile_vector, dsts[2].header.object);
            TEST_ASSERT_EQ(mtxfile_coordinate, dsts[2].header.format);
            TEST_ASSERT_EQ(mtxfile_real, dsts[2].header.field);
            TEST_ASSERT_EQ(mtxfile_general, dsts[2].header.symmetry);
            TEST_ASSERT_EQ(mtx_double, dsts[2].precision);
            TEST_ASSERT_EQ(8, dsts[2].size.num_rows);
            TEST_ASSERT_EQ(-1, dsts[2].size.num_columns);
            TEST_ASSERT_EQ(1, dsts[2].size.num_nonzeros);
            const struct mtxfile_vector_coordinate_real_double * data =
                dsts[2].data.vector_coordinate_real_double;
            TEST_ASSERT_EQ(8, data[0].i); TEST_ASSERT_EQ(8.0, data[0].a);
            mtxfile_free(&dsts[2]);
        }
        mtxfile_free(&src);
    }
    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_permute()’ tests permuting rows and columns of
 * matrices and vectors in Matrix Market format.
 */
int test_mtxfile_permute(void)
{
    int err;

    /*
     * Array formats.
     */

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 3};
        err = mtxfile_permute(&mtx, permutation, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(4.0, data[0]);
        TEST_ASSERT_EQ(5.0, data[1]);
        TEST_ASSERT_EQ(6.0, data[2]);
        TEST_ASSERT_EQ(1.0, data[3]);
        TEST_ASSERT_EQ(2.0, data[4]);
        TEST_ASSERT_EQ(3.0, data[5]);
        TEST_ASSERT_EQ(7.0, data[6]);
        TEST_ASSERT_EQ(8.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 3};
        err = mtxfile_permute(&mtx, NULL, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(2.0, data[0]);
        TEST_ASSERT_EQ(1.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        TEST_ASSERT_EQ(5.0, data[3]);
        TEST_ASSERT_EQ(4.0, data[4]);
        TEST_ASSERT_EQ(6.0, data[5]);
        TEST_ASSERT_EQ(8.0, data[6]);
        TEST_ASSERT_EQ(7.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        int num_columns = 3;
        const double srcdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        struct mtxfile mtx;
        err = mtxfile_init_matrix_array_real_double(
            &mtx, mtxfile_general, num_rows, num_columns, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 3};
        err = mtxfile_permute(&mtx, permutation, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(3, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(5.0, data[0]);
        TEST_ASSERT_EQ(4.0, data[1]);
        TEST_ASSERT_EQ(6.0, data[2]);
        TEST_ASSERT_EQ(2.0, data[3]);
        TEST_ASSERT_EQ(1.0, data[4]);
        TEST_ASSERT_EQ(3.0, data[5]);
        TEST_ASSERT_EQ(8.0, data[6]);
        TEST_ASSERT_EQ(7.0, data[7]);
        TEST_ASSERT_EQ(9.0, data[8]);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 3;
        const double srcdata[] = {1.0, 2.0, 3.0};
        struct mtxfile mtx;
        err = mtxfile_init_vector_array_real_double(
            &mtx, num_rows, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 3};
        err = mtxfile_permute(&mtx, permutation, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_array, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_double, mtx.precision);
        TEST_ASSERT_EQ(3, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(-1, mtx.size.num_nonzeros);
        const double * data = mtx.data.array_real_double;
        TEST_ASSERT_EQ(2.0, data[0]);
        TEST_ASSERT_EQ(1.0, data[1]);
        TEST_ASSERT_EQ(3.0, data[2]);
        mtxfile_free(&mtx);
    }

    /*
     * Matrix coordinate formats.
     */

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1,1.0f}, {1,4,2.0f},
            {2,2,3.0f},
            {3,3,4.0f},
            {4,1,5.0f}, {4,4,6.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 4, 3};
        err = mtxfile_permute(&mtx, permutation, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(   2, data[0].i); TEST_ASSERT_EQ(   1, data[0].j);
        TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(   2, data[1].i); TEST_ASSERT_EQ(   4, data[1].j);
        TEST_ASSERT_EQ(2.0f, data[1].a);
        TEST_ASSERT_EQ(   1, data[2].i); TEST_ASSERT_EQ(   2, data[2].j);
        TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(   4, data[3].i); TEST_ASSERT_EQ(   3, data[3].j);
        TEST_ASSERT_EQ(4.0f, data[3].a);
        TEST_ASSERT_EQ(   3, data[4].i); TEST_ASSERT_EQ(   1, data[4].j);
        TEST_ASSERT_EQ(5.0f, data[4].a);
        TEST_ASSERT_EQ(   3, data[5].i); TEST_ASSERT_EQ(   4, data[5].j);
        TEST_ASSERT_EQ(6.0f, data[5].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1,1.0f}, {1,4,2.0f},
            {2,2,3.0f},
            {3,3,4.0f},
            {4,1,5.0f}, {4,4,6.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 4, 3};
        err = mtxfile_permute(&mtx, NULL, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(   1, data[0].i); TEST_ASSERT_EQ(   2, data[0].j);
        TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(   1, data[1].i); TEST_ASSERT_EQ(   3, data[1].j);
        TEST_ASSERT_EQ(2.0f, data[1].a);
        TEST_ASSERT_EQ(   2, data[2].i); TEST_ASSERT_EQ(   1, data[2].j);
        TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(   3, data[3].i); TEST_ASSERT_EQ(   4, data[3].j);
        TEST_ASSERT_EQ(4.0f, data[3].a);
        TEST_ASSERT_EQ(   4, data[4].i); TEST_ASSERT_EQ(   2, data[4].j);
        TEST_ASSERT_EQ(5.0f, data[4].a);
        TEST_ASSERT_EQ(   4, data[5].i); TEST_ASSERT_EQ(   3, data[5].j);
        TEST_ASSERT_EQ(6.0f, data[5].a);
        mtxfile_free(&mtx);
    }

    {
        int num_rows = 4;
        int num_columns = 4;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1,1.0f}, {1,4,2.0f},
            {2,2,3.0f},
            {3,3,4.0f},
            {4,1,5.0f}, {4,4,6.0f}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 4, 3};
        err = mtxfile_permute(&mtx, permutation, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(4, mtx.size.num_columns);
        TEST_ASSERT_EQ(6, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtx.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(   2, data[0].i); TEST_ASSERT_EQ(   2, data[0].j);
        TEST_ASSERT_EQ(1.0f, data[0].a);
        TEST_ASSERT_EQ(   2, data[1].i); TEST_ASSERT_EQ(   3, data[1].j);
        TEST_ASSERT_EQ(2.0f, data[1].a);
        TEST_ASSERT_EQ(   1, data[2].i); TEST_ASSERT_EQ(   1, data[2].j);
        TEST_ASSERT_EQ(3.0f, data[2].a);
        TEST_ASSERT_EQ(   4, data[3].i); TEST_ASSERT_EQ(   4, data[3].j);
        TEST_ASSERT_EQ(4.0f, data[3].a);
        TEST_ASSERT_EQ(   3, data[4].i); TEST_ASSERT_EQ(   2, data[4].j);
        TEST_ASSERT_EQ(5.0f, data[4].a);
        TEST_ASSERT_EQ(   3, data[5].i); TEST_ASSERT_EQ(   3, data[5].j);
        TEST_ASSERT_EQ(6.0f, data[5].a);
        mtxfile_free(&mtx);
    }

    {
        /* Permute rows and columns.    */
        /*                              */
        /*    5--7--6           7--8--9 */
        /*    |  | /            |  | /  */
        /* 4--8--2     -->   3--5--6    */
        /* |  |  |           |  |  |    */
        /* 9--1--3           1--2--4    */

        int num_rows = 9;
        int num_columns = 9;
        const struct mtxfile_matrix_coordinate_pattern srcdata[] = {
            {1,3}, {1,8}, {1,9},
            {2,3}, {2,6}, {2,7}, {2,8},
            {3,1}, {3,2},
            {4,8}, {4,9},
            {5,7}, {5,8},
            {6,2}, {6,7},
            {7,2}, {7,5}, {7,6},
            {8,1}, {8,2}, {8,4}, {8,5},
            {9,1}, {9,4}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_matrix_coordinate_pattern(
            &mtx, mtxfile_general, num_rows, num_columns, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2,6,4,3,7,9,8,5,1};
        err = mtxfile_permute(&mtx, permutation, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        err = mtxfile_sort(&mtx, mtxfile_row_major, num_nonzeros, NULL);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_matrix, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_pattern, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(9, mtx.size.num_rows);
        TEST_ASSERT_EQ(9, mtx.size.num_columns);
        TEST_ASSERT_EQ(24, mtx.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_pattern * data =
            mtx.data.matrix_coordinate_pattern;
        TEST_ASSERT_EQ(1, data[ 0].i); TEST_ASSERT_EQ(2, data[ 0].j);
        TEST_ASSERT_EQ(1, data[ 1].i); TEST_ASSERT_EQ(3, data[ 1].j);
        TEST_ASSERT_EQ(2, data[ 2].i); TEST_ASSERT_EQ(1, data[ 2].j);
        TEST_ASSERT_EQ(2, data[ 3].i); TEST_ASSERT_EQ(4, data[ 3].j);
        TEST_ASSERT_EQ(2, data[ 4].i); TEST_ASSERT_EQ(5, data[ 4].j);
        TEST_ASSERT_EQ(3, data[ 5].i); TEST_ASSERT_EQ(1, data[ 5].j);
        TEST_ASSERT_EQ(3, data[ 6].i); TEST_ASSERT_EQ(5, data[ 6].j);
        TEST_ASSERT_EQ(4, data[ 7].i); TEST_ASSERT_EQ(2, data[ 7].j);
        TEST_ASSERT_EQ(4, data[ 8].i); TEST_ASSERT_EQ(6, data[ 8].j);
        TEST_ASSERT_EQ(5, data[ 9].i); TEST_ASSERT_EQ(2, data[ 9].j);
        TEST_ASSERT_EQ(5, data[10].i); TEST_ASSERT_EQ(3, data[10].j);
        TEST_ASSERT_EQ(5, data[11].i); TEST_ASSERT_EQ(6, data[11].j);
        TEST_ASSERT_EQ(5, data[12].i); TEST_ASSERT_EQ(7, data[12].j);
        TEST_ASSERT_EQ(6, data[13].i); TEST_ASSERT_EQ(4, data[13].j);
        TEST_ASSERT_EQ(6, data[14].i); TEST_ASSERT_EQ(5, data[14].j);
        TEST_ASSERT_EQ(6, data[15].i); TEST_ASSERT_EQ(8, data[15].j);
        TEST_ASSERT_EQ(6, data[16].i); TEST_ASSERT_EQ(9, data[16].j);
        TEST_ASSERT_EQ(7, data[17].i); TEST_ASSERT_EQ(5, data[17].j);
        TEST_ASSERT_EQ(7, data[18].i); TEST_ASSERT_EQ(8, data[18].j);
        TEST_ASSERT_EQ(8, data[19].i); TEST_ASSERT_EQ(6, data[19].j);
        TEST_ASSERT_EQ(8, data[20].i); TEST_ASSERT_EQ(7, data[20].j);
        TEST_ASSERT_EQ(8, data[21].i); TEST_ASSERT_EQ(9, data[21].j);
        TEST_ASSERT_EQ(9, data[22].i); TEST_ASSERT_EQ(6, data[22].j);
        TEST_ASSERT_EQ(9, data[23].i); TEST_ASSERT_EQ(8, data[23].j);
        mtxfile_free(&mtx);
    }

    /*
     * Vector coordinate formats.
     */

    {
        int num_rows = 4;
        const struct mtxfile_vector_coordinate_integer_single srcdata[] = {
            {3,4}, {1,2}, {4,5}};
        int64_t num_nonzeros = sizeof(srcdata) / sizeof(*srcdata);
        struct mtxfile mtx;
        err = mtxfile_init_vector_coordinate_integer_single(
            &mtx, num_rows, num_nonzeros, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        const int permutation[] = {2, 1, 4, 3};
        err = mtxfile_permute(&mtx, permutation, permutation);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(mtxfile_vector, mtx.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtx.header.format);
        TEST_ASSERT_EQ(mtxfile_integer, mtx.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtx.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtx.precision);
        TEST_ASSERT_EQ(4, mtx.size.num_rows);
        TEST_ASSERT_EQ(-1, mtx.size.num_columns);
        TEST_ASSERT_EQ(3, mtx.size.num_nonzeros);
        const struct mtxfile_vector_coordinate_integer_single * data =
            mtx.data.vector_coordinate_integer_single;
        TEST_ASSERT_EQ(4, data[0].i); TEST_ASSERT_EQ(4, data[0].a);
        TEST_ASSERT_EQ(2, data[1].i); TEST_ASSERT_EQ(2, data[1].a);
        TEST_ASSERT_EQ(3, data[2].i); TEST_ASSERT_EQ(5, data[2].a);
        mtxfile_free(&mtx);
    }

    return TEST_SUCCESS;
}

/**
 * ‘test_mtxfile_reorder_rcm()’ tests reordering a matrix in
 * coordinate format using the Reverse Cuthill-McKee algorithm.
 */
int test_mtxfile_reorder_rcm(void)
{
    {
        /* Symmetric matrix */

        /* 1--3--5        5--4--1 */
        /* |  |     -->   |  |    */
        /* 2--4           3--2    */

        int err;
        struct mtxfile mtxfile;
        int num_rows = 5;
        int num_columns = 5;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,2, 1.0f},
            {1,3, 2.0f},
            {2,1, 3.0f},
            {2,4, 4.0f},
            {3,1, 5.0f},
            {3,4, 6.0f},
            {3,5, 7.0f},
            {4,2, 8.0f},
            {4,3, 9.0f},
            {5,3,10.0f}};
        size_t size = sizeof(srcdata) / sizeof(*srcdata);
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        bool symmetric;
        int starting_vertex = 1;
        int rowperm[5] = {};
        int colperm[5] = {};
        err = mtxfile_reorder_rcm(
            &mtxfile, rowperm, colperm, true, &symmetric, &starting_vertex);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(true, symmetric);
        TEST_ASSERT_EQ(5, rowperm[0]);
        TEST_ASSERT_EQ(3, rowperm[1]);
        TEST_ASSERT_EQ(4, rowperm[2]);
        TEST_ASSERT_EQ(2, rowperm[3]);
        TEST_ASSERT_EQ(1, rowperm[4]);

        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        TEST_ASSERT_EQ(5, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(5, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(10, mtxfile.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    5, data[0].i); TEST_ASSERT_EQ(3, data[0].j);
        TEST_ASSERT_EQ( 1.0f, data[0].a);
        TEST_ASSERT_EQ(    5, data[1].i); TEST_ASSERT_EQ(4, data[1].j);
        TEST_ASSERT_EQ( 2.0f, data[1].a);
        TEST_ASSERT_EQ(    3, data[2].i); TEST_ASSERT_EQ(5, data[2].j);
        TEST_ASSERT_EQ( 3.0f, data[2].a);
        TEST_ASSERT_EQ(    3, data[3].i); TEST_ASSERT_EQ(2, data[3].j);
        TEST_ASSERT_EQ( 4.0f, data[3].a);
        TEST_ASSERT_EQ(    4, data[4].i); TEST_ASSERT_EQ(5, data[4].j);
        TEST_ASSERT_EQ( 5.0f, data[4].a);
        TEST_ASSERT_EQ(    4, data[5].i); TEST_ASSERT_EQ(2, data[5].j);
        TEST_ASSERT_EQ( 6.0f, data[5].a);
        TEST_ASSERT_EQ(    4, data[6].i); TEST_ASSERT_EQ(1, data[6].j);
        TEST_ASSERT_EQ( 7.0f, data[6].a);
        TEST_ASSERT_EQ(    2, data[7].i); TEST_ASSERT_EQ(3, data[7].j);
        TEST_ASSERT_EQ( 8.0f, data[7].a);
        TEST_ASSERT_EQ(    2, data[8].i); TEST_ASSERT_EQ(4, data[8].j);
        TEST_ASSERT_EQ( 9.0f, data[8].a);
        TEST_ASSERT_EQ(    1, data[9].i); TEST_ASSERT_EQ(4, data[9].j);
        TEST_ASSERT_EQ(10.0f, data[9].a);
        mtxfile_free(&mtxfile);
    }

    {
        /* Unsymmetric matrix */

        /* 1<-3<-5        5<-4<-1 */
        /* ↑  ↑     ==>   ↑  ↑    */
        /* 2->4           3->2    */

        int err;
        struct mtxfile mtxfile;
        int num_rows = 5;
        int num_columns = 5;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {2,1, 1.0f},
            {3,1, 2.0f},
            {4,2, 4.0f},
            {4,3, 6.0f},
            {5,3, 7.0f}};
        size_t size = sizeof(srcdata) / sizeof(*srcdata);
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        bool symmetric;
        int starting_vertex = 1;
        int rowperm[5] = {};
        int colperm[5] = {};
        err = mtxfile_reorder_rcm(
            &mtxfile, rowperm, colperm, true, &symmetric, &starting_vertex);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(true, symmetric);
        TEST_ASSERT_EQ(5, rowperm[0]);
        TEST_ASSERT_EQ(3, rowperm[1]);
        TEST_ASSERT_EQ(4, rowperm[2]);
        TEST_ASSERT_EQ(2, rowperm[3]);
        TEST_ASSERT_EQ(1, rowperm[4]);

        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        TEST_ASSERT_EQ(5, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(5, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(5, mtxfile.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    3, data[0].i); TEST_ASSERT_EQ(5, data[0].j);
        TEST_ASSERT_EQ( 1.0f, data[0].a);
        TEST_ASSERT_EQ(    4, data[1].i); TEST_ASSERT_EQ(5, data[1].j);
        TEST_ASSERT_EQ( 2.0f, data[1].a);
        TEST_ASSERT_EQ(    2, data[2].i); TEST_ASSERT_EQ(3, data[2].j);
        TEST_ASSERT_EQ( 4.0f, data[2].a);
        TEST_ASSERT_EQ(    2, data[3].i); TEST_ASSERT_EQ(4, data[3].j);
        TEST_ASSERT_EQ( 6.0f, data[3].a);
        TEST_ASSERT_EQ(    1, data[4].i); TEST_ASSERT_EQ(4, data[4].j);
        TEST_ASSERT_EQ( 7.0f, data[4].a);
        mtxfile_free(&mtxfile);
    }

    {
        /* Disconnected graph: */

        /* 1  3<-5        5  3<-1 */
        /* ↑  ↑     ==>   ↑  ↑    */
        /* 2  4           4  2    */

        int err;
        struct mtxfile mtxfile;
        int num_rows = 5;
        int num_columns = 5;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {2,1, 1.0f},
            {4,3, 6.0f},
            {5,3, 7.0f}};
        size_t size = sizeof(srcdata) / sizeof(*srcdata);
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        bool symmetric;
        int starting_vertex = 1;
        int rowperm[5] = {};
        int colperm[5] = {};
        err = mtxfile_reorder_rcm(
            &mtxfile, rowperm, colperm, true, &symmetric, &starting_vertex);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(true, symmetric);
        TEST_ASSERT_EQ(5, rowperm[0]);
        TEST_ASSERT_EQ(4, rowperm[1]);
        TEST_ASSERT_EQ(3, rowperm[2]);
        TEST_ASSERT_EQ(2, rowperm[3]);
        TEST_ASSERT_EQ(1, rowperm[4]);

        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        TEST_ASSERT_EQ(5, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(5, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(3, mtxfile.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    4, data[0].i); TEST_ASSERT_EQ(5, data[0].j);
        TEST_ASSERT_EQ( 1.0f, data[0].a);
        TEST_ASSERT_EQ(    2, data[1].i); TEST_ASSERT_EQ(3, data[1].j);
        TEST_ASSERT_EQ( 6.0f, data[1].a);
        TEST_ASSERT_EQ(    1, data[2].i); TEST_ASSERT_EQ(3, data[2].j);
        TEST_ASSERT_EQ( 7.0f, data[2].a);
        mtxfile_free(&mtxfile);
    }

    {
        /* Rectangular matrix: */

        /* Rows:      3 2 1      1 2 3 */
        /*            |/|/   ==> |/|/  */
        /* Columns:   2 1        1 2   */

        int err;
        struct mtxfile mtxfile;
        int num_rows = 3;
        int num_columns = 2;
        const struct mtxfile_matrix_coordinate_real_single srcdata[] = {
            {1,1, 1.0f},
            {2,1, 2.0f},
            {2,2, 3.0f},
            {3,2, 4.0f}};
        size_t size = sizeof(srcdata) / sizeof(*srcdata);
        err = mtxfile_init_matrix_coordinate_real_single(
            &mtxfile, mtxfile_general, num_rows, num_columns, size, srcdata);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));

        bool symmetric;
        int starting_vertex = 1;
        int rowperm[3] = {};
        int colperm[2] = {};
        err = mtxfile_reorder_rcm(
            &mtxfile, rowperm, colperm, true, &symmetric, &starting_vertex);
        TEST_ASSERT_EQ(false, symmetric);
        TEST_ASSERT_EQ_MSG(MTX_SUCCESS, err, "%s", mtxstrerror(err));
        TEST_ASSERT_EQ(3, rowperm[0]);
        TEST_ASSERT_EQ(2, rowperm[1]);
        TEST_ASSERT_EQ(1, rowperm[2]);
        TEST_ASSERT_EQ(2, colperm[0]);
        TEST_ASSERT_EQ(1, colperm[1]);

        TEST_ASSERT_EQ(mtxfile_matrix, mtxfile.header.object);
        TEST_ASSERT_EQ(mtxfile_coordinate, mtxfile.header.format);
        TEST_ASSERT_EQ(mtxfile_real, mtxfile.header.field);
        TEST_ASSERT_EQ(mtxfile_general, mtxfile.header.symmetry);
        TEST_ASSERT_EQ(mtx_single, mtxfile.precision);
        TEST_ASSERT_EQ(3, mtxfile.size.num_rows);
        TEST_ASSERT_EQ(2, mtxfile.size.num_columns);
        TEST_ASSERT_EQ(4, mtxfile.size.num_nonzeros);
        const struct mtxfile_matrix_coordinate_real_single * data =
            mtxfile.data.matrix_coordinate_real_single;
        TEST_ASSERT_EQ(    3, data[0].i); TEST_ASSERT_EQ(2, data[0].j);
        TEST_ASSERT_EQ( 1.0f, data[0].a);
        TEST_ASSERT_EQ(    2, data[1].i); TEST_ASSERT_EQ(2, data[1].j);
        TEST_ASSERT_EQ( 2.0f, data[1].a);
        TEST_ASSERT_EQ(    2, data[2].i); TEST_ASSERT_EQ(1, data[2].j);
        TEST_ASSERT_EQ( 3.0f, data[2].a);
        TEST_ASSERT_EQ(    1, data[3].i); TEST_ASSERT_EQ(1, data[3].j);
        TEST_ASSERT_EQ( 4.0f, data[3].a);
        mtxfile_free(&mtxfile);
    }
    return TEST_SUCCESS;
}

/**
 * ‘main()’ entry point and test driver.
 */
int main(int argc, char * argv[])
{
    TEST_SUITE_BEGIN("Running tests for Matrix Market files\n");
    TEST_RUN(test_mtxfileheader_parse);
    TEST_RUN(test_mtxfilesize_parse);
    TEST_RUN(test_mtxfiledata_parse);
    TEST_RUN(test_mtxfileheader_fread);
    TEST_RUN(test_mtxfile_fread_comments);
    TEST_RUN(test_mtxfilesize_fread);
    TEST_RUN(test_mtxfiledata_fread);
    TEST_RUN(test_mtxfile_fread);
    TEST_RUN(test_mtxfile_fwrite);
#ifdef LIBMTX_HAVE_LIBZ
    TEST_RUN(test_mtxfile_gzread);
    TEST_RUN(test_mtxfile_gzwrite);
#endif
    TEST_RUN(test_mtxfile_transpose);
    TEST_RUN(test_mtxfile_sort);
    TEST_RUN(test_mtxfile_compact);
    TEST_RUN(test_mtxfile_assemble);
    TEST_RUN(test_mtxfile_partition_nonzeros);
    TEST_RUN(test_mtxfile_partition_rowwise);
    TEST_RUN(test_mtxfile_partition_columnwise);
    TEST_RUN(test_mtxfile_partition_2d);
    TEST_RUN(test_mtxfile_split);
    TEST_RUN(test_mtxfile_permute);
    TEST_RUN(test_mtxfile_reorder_rcm);
    TEST_SUITE_END();
    return (TEST_SUITE_STATUS == TEST_SUCCESS) ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
