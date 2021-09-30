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
 * Last modified: 2021-09-30
 *
 * Data structures for distributed vectors.
 */

#include <libmtx/libmtx-config.h>

#ifdef LIBMTX_HAVE_MPI
#include <libmtx/error.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/mtxfile/header.h>
#include <libmtx/mtxfile/size.h>
#include <libmtx/util/field.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/distvector.h>
#include <libmtx/vector/vector.h>

#include <mpi.h>

#ifdef LIBMTX_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Memory management
 */

/**
 * ‘mtxdistvector_free()’ frees storage allocated for a vector.
 */
void mtxdistvector_free(
    struct mtxdistvector * distvector)
{
    mtx_partition_free(&distvector->partition);
    mtxvector_free(&distvector->interior);
}

/**
 * ‘mtxdistvector_alloc_copy()’ allocates storage for a copy of a
 * distributed vector without initialising the underlying values.
 */
int mtxdistvector_alloc_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src);

/**
 * ‘mtxdistvector_init_copy()’ creates a copy of a distributed vector.
 */
int mtxdistvector_init_copy(
    struct mtxdistvector * dst,
    const struct mtxdistvector * src);

/*
 * Distributed vectors in array format
 */

/**
 * ‘mtxdistvector_alloc_array()’ allocates a distributed vector in
 * array format.
 */
int mtxdistvector_alloc_array(
    struct mtxdistvector * vector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int size);

/**
 * ‘mtxdistvector_init_array_real_single()’ allocates and initialises
 * a distributed vector in array format with real, single precision
 * coefficients.
 */
int mtxdistvector_init_array_real_single(
    struct mtxdistvector * vector,
    int size,
    const float * data);

/**
 * ‘mtxdistvector_init_array_real_double()’ allocates and initialises
 * a distributed vector in array format with real, double precision
 * coefficients.
 */
int mtxdistvector_init_array_real_double(
    struct mtxdistvector * vector,
    int size,
    const double * data);

/**
 * ‘mtxdistvector_init_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_array_complex_single(
    struct mtxdistvector * vector,
    int size,
    const float (* data)[2]);

/**
 * ‘mtxdistvector_init_array_complex_double()’ allocates and
 * initialises a distributed vector in array format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_array_complex_double(
    struct mtxdistvector * vector,
    int size,
    const double (* data)[2]);

/**
 * ‘mtxdistvector_init_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_array_integer_single(
    struct mtxdistvector * vector,
    int size,
    const int32_t * data);

/**
 * ‘mtxdistvector_init_array_integer_double()’ allocates and
 * initialises a distributed vector in array format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_array_integer_double(
    struct mtxdistvector * vector,
    int size,
    const int64_t * data);

/*
 * Convert to and from Matrix Market format
 */

/**
 * ‘mtxdistvector_from_mtxfile()’ converts a vector in Matrix Market
 * format to a distributed vector.
 */
int mtxdistvector_from_mtxfile(
    struct mtxdistvector * distvector,
    const struct mtxfile * mtxfile,
    MPI_Comm comm,
    int root,
    struct mtxmpierror * mpierror)
{
    int err;
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;

    int comm_size;
    mpierror->mpierrcode = MPI_Comm_size(comm, &comm_size);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    int rank;
    mpierror->mpierrcode = MPI_Comm_rank(comm, &rank);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;

    distvector->comm = comm;
    distvector->comm_size = comm_size;
    distvector->rank = rank;

    /* 1. Distribute the Matrix Market file among processes. */
    struct mtxdistfile src;
    err = mtxdistfile_from_mtxfile(&src, mtxfile, comm, root, mpierror);
    if (err)
        return err;

    /* 2. Partition the rows of the vector. */
    enum mtx_partition_type partition_type = mtx_block;
    int num_parts = comm_size;
    err = mtx_partition_init(
        &distvector->partition, partition_type,
        src.size.num_rows, num_parts, 0, NULL);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 3. Partition and redistribute the Matrix Market file. */
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&src.mtxfile.size, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtx_partition_free(&distvector->partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * part_per_data_line = malloc(num_data_lines * sizeof(int));
    err = !part_per_data_line ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtx_partition_free(&distvector->partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * data_lines_per_part_ptr = malloc((num_parts+1) * sizeof(int64_t));
    err = !data_lines_per_part_ptr ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(part_per_data_line);
        mtx_partition_free(&distvector->partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * data_lines_per_part = malloc(num_data_lines * sizeof(int64_t));
    err = !data_lines_per_part ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(data_lines_per_part_ptr);
        free(part_per_data_line);
        mtx_partition_free(&distvector->partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxdistfile_partition_rows(
        &src, &distvector->partition, part_per_data_line,
        data_lines_per_part_ptr, data_lines_per_part, mpierror);
    if (err) {
        free(data_lines_per_part);
        free(data_lines_per_part_ptr);
        free(part_per_data_line);
        mtx_partition_free(&distvector->partition);
        mtxdistfile_free(&src);
        return err;
    }

    struct mtxdistfile dst;
    err = mtxdistfile_init_from_partition(
        &dst, &src, num_parts, data_lines_per_part_ptr, data_lines_per_part, mpierror);
    if (err) {
        free(data_lines_per_part);
        free(data_lines_per_part_ptr);
        free(part_per_data_line);
        mtx_partition_free(&distvector->partition);
        mtxdistfile_free(&src);
        return err;
    }
    free(data_lines_per_part);
    free(data_lines_per_part_ptr);
    free(part_per_data_line);
    mtxdistfile_free(&src);

    /* 4. Create the distributed vector. */
    err = mtxvector_from_mtxfile(
        &distvector->interior, &dst.mtxfile, mtxvector_auto);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(&dst);
        mtx_partition_free(&distvector->partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxdistfile_free(&dst);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_to_mtxfile()’ converts a distributed vector to a
 * vector in Matrix Market format.
 */
int mtxdistvector_to_mtxfile(
    const struct mtxdistvector * vector,
    struct mtxfile * mtxfile);

/*
 * I/O functions
 */

/**
 * ‘mtxdistvector_read()’ reads a vector from a Matrix Market file.
 * The file may optionally be compressed by gzip.
 *
 * The ‘precision’ argument specifies which precision to use for
 * storing matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_read(
    struct mtxdistvector * vector,
    enum mtx_precision precision,
    const char * path,
    bool gzip,
    int * lines_read,
    int64_t * bytes_read);

/**
 * ‘mtxdistvector_fread()’ reads a vector from a stream in Matrix
 * Market format.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_fread(
    struct mtxdistvector * vector,
    enum mtx_precision precision,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistvector_gzread()’ reads a vector from a gzip-compressed
 * stream.
 *
 * ‘precision’ is used to determine the precision to use for storing
 * the values of matrix or vector entries.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the vector.
 */
int mtxdistvector_gzread(
    struct mtxdistvector * vector,
    enum mtx_precision precision,
    gzFile f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);
#endif

/**
 * ‘mtxdistvector_write()’ writes a vector to a Matrix Market
 * file. The file may optionally be compressed by gzip.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘format’ is ‘NULL’, then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 */
int mtxdistvector_write(
    const struct mtxdistvector * vector,
    const char * path,
    bool gzip,
    const char * format,
    int64_t * bytes_written);

/**
 * ‘mtxdistvector_fwrite()’ writes a vector to a stream.
 *
 * If ‘format’ is ‘NULL’, then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxdistvector_fwrite(
    const struct mtxdistvector * vector,
    FILE * f,
    const char * format,
    int64_t * bytes_written);

#ifdef LIBMTX_HAVE_LIBZ
/**
 * ‘mtxdistvector_gzwrite()’ writes a vector to a gzip-compressed
 * stream.
 *
 * If ‘format’ is ‘NULL’, then the format specifier '%d' is used to
 * print integers and '%f' is used to print floating point
 * numbers. Otherwise, the given format string is used when printing
 * numerical values.
 *
 * The format string follows the conventions of ‘printf’. If the field
 * is ‘real’, ‘double’ or ‘complex’, then the format specifiers '%e',
 * '%E', '%f', '%F', '%g' or '%G' may be used. If the field is
 * ‘integer’, then the format specifier must be '%d'. The format
 * string is ignored if the field is ‘pattern’. Field width and
 * precision may be specified (e.g., "%3.1f"), but variable field
 * width and precision (e.g., "%*.*f"), as well as length modifiers
 * (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxdistvector_gzwrite(
    const struct mtxdistvector * vector,
    gzFile f,
    const char * format,
    int64_t * bytes_written);
#endif
#endif
