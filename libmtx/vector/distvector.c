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

#include <math.h>
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

static int mtxdistvector_init_comm(
    struct mtxdistvector * distvector,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
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
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_alloc_array()’ allocates a distributed vector in
 * array format.
 */
int mtxdistvector_alloc_array(
    struct mtxdistvector * distvector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int size,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_alloc_array(
        &distvector->interior, field, precision, size);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_real_single()’ allocates and initialises
 * a distributed vector in array format with real, single precision
 * coefficients.
 */
int mtxdistvector_init_array_real_single(
    struct mtxdistvector * distvector,
    int size,
    const float * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_array_real_single(
        &distvector->interior, size, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_real_double()’ allocates and initialises
 * a distributed vector in array format with real, double precision
 * coefficients.
 */
int mtxdistvector_init_array_real_double(
    struct mtxdistvector * distvector,
    int size,
    const double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_array_real_double(
        &distvector->interior, size, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_complex_single()’ allocates and
 * initialises a distributed vector in array format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_array_complex_single(
    struct mtxdistvector * distvector,
    int size,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_array_complex_single(
        &distvector->interior, size, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_complex_double()’ allocates and
 * initialises a distributed vector in array format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_array_complex_double(
    struct mtxdistvector * distvector,
    int size,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_array_complex_double(
        &distvector->interior, size, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_integer_single()’ allocates and
 * initialises a distributed vector in array format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_array_integer_single(
    struct mtxdistvector * distvector,
    int size,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_array_integer_single(
        &distvector->interior, size, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_array_integer_double()’ allocates and
 * initialises a distributed vector in array format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_array_integer_double(
    struct mtxdistvector * distvector,
    int size,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_array_integer_double(
        &distvector->interior, size, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/*
 * Distributed vectors in coordinate format
 */

/**
 * ‘mtxdistvector_alloc_coordinate()’ allocates a distributed vector
 * in coordinate format.
 */
int mtxdistvector_alloc_coordinate(
    struct mtxdistvector * distvector,
    enum mtx_field_ field,
    enum mtx_precision precision,
    int size,
    int64_t num_nonzeros,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_alloc_coordinate(
        &distvector->interior, field, precision, size, num_nonzeros);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_real_single()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_real_single(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const float * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_real_single(
        &distvector->interior, size, num_nonzeros, indices, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_real_double()’ allocates and
 * initialises a distributed vector in coordinate format with real,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_real_double(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const double * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_real_double(
        &distvector->interior, size, num_nonzeros, indices, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_complex_single()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_single(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const float (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_complex_single(
        &distvector->interior, size, num_nonzeros, indices, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_complex_double()’ allocates and
 * initialises a distributed vector in coordinate format with complex,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_complex_double(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const double (* data)[2],
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_complex_double(
        &distvector->interior, size, num_nonzeros, indices, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_integer_single()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * single precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_single(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const int32_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_integer_single(
        &distvector->interior, size, num_nonzeros, indices, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_integer_double()’ allocates and
 * initialises a distributed vector in coordinate format with integer,
 * double precision coefficients.
 */
int mtxdistvector_init_coordinate_integer_double(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    const int64_t * data,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_integer_double(
        &distvector->interior, size, num_nonzeros, indices, data);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_init_coordinate_pattern()’ allocates and initialises
 * a distributed vector in coordinate format with boolean
 * coefficients.
 */
int mtxdistvector_init_coordinate_pattern(
    struct mtxdistvector * distvector,
    int size,
    int64_t num_nonzeros,
    const int * indices,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_init_coordinate_pattern(
        &distvector->interior, size, num_nonzeros, indices);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

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
    enum mtxvector_type vector_type,
    MPI_Comm comm,
    int root,
    struct mtxmpierror * mpierror)
{
    int err;
    if (mtxfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;

    /* 1. Distribute the Matrix Market file among processes. */
    struct mtxdistfile src;
    err = mtxdistfile_from_mtxfile(&src, mtxfile, comm, root, mpierror);
    if (err)
        return err;

    /* 2. Partition the rows of the vector. */
    struct mtx_partition row_partition;
    enum mtx_partition_type partition_type = mtx_block;
    int num_parts = distvector->comm_size;
    err = mtx_partition_init(
        &row_partition, partition_type,
        src.size.num_rows, num_parts, 0, NULL);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }

    /* 3. Partition and redistribute the Matrix Market file. */
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&src.mtxfile.size, &num_data_lines);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtx_partition_free(&row_partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int * part_per_data_line = malloc(num_data_lines * sizeof(int));
    err = !part_per_data_line ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtx_partition_free(&row_partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * data_lines_per_part_ptr = malloc((num_parts+1) * sizeof(int64_t));
    err = !data_lines_per_part_ptr ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(part_per_data_line);
        mtx_partition_free(&row_partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    int64_t * data_lines_per_part = malloc(num_data_lines * sizeof(int64_t));
    err = !data_lines_per_part ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err)) {
        free(data_lines_per_part_ptr);
        free(part_per_data_line);
        mtx_partition_free(&row_partition);
        mtxdistfile_free(&src);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    err = mtxdistfile_partition_rows(
        &src, &row_partition, part_per_data_line,
        data_lines_per_part_ptr, data_lines_per_part, mpierror);
    if (err) {
        free(data_lines_per_part);
        free(data_lines_per_part_ptr);
        free(part_per_data_line);
        mtx_partition_free(&row_partition);
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
        mtx_partition_free(&row_partition);
        mtxdistfile_free(&src);
        return err;
    }
    free(data_lines_per_part);
    free(data_lines_per_part_ptr);
    free(part_per_data_line);
    mtxdistfile_free(&src);

    /* 4. Create the distributed vector. */
    err = mtxvector_from_mtxfile(
        &distvector->interior, &dst.mtxfile, vector_type);
    if (mtxmpierror_allreduce(mpierror, err)) {
        mtxdistfile_free(&dst);
        mtx_partition_free(&row_partition);
        return MTX_ERR_MPI_COLLECTIVE;
    }
    mtxdistfile_free(&dst);
    mtx_partition_free(&row_partition);
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_from_mtxdistfile()’ converts a vector in distributed
 * Matrix Market format to a distributed vector.
 */
int mtxdistvector_from_mtxdistfile(
    struct mtxdistvector * distvector,
    const struct mtxdistfile * mtxdistfile,
    enum mtxvector_type vector_type,
    MPI_Comm comm,
    struct mtxmpierror * mpierror)
{
    int err;
    if (mtxdistfile->header.object != mtxfile_vector)
        return MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    err = mtxdistvector_init_comm(distvector, comm, mpierror);
    if (err)
        return err;
    err = mtxvector_from_mtxfile(
        &distvector->interior, &mtxdistfile->mtxfile, vector_type);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * ‘mtxdistvector_to_mtxfile()’ converts a distributed vector to a
 * vector in Matrix Market format.
 */
int mtxdistvector_to_mtxfile(
    const struct mtxdistvector * distvector,
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
    struct mtxdistvector * distvector,
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
    struct mtxdistvector * distvector,
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
    struct mtxdistvector * distvector,
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
    const struct mtxdistvector * distvector,
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
    const struct mtxdistvector * distvector,
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
    const struct mtxdistvector * distvector,
    gzFile f,
    const char * format,
    int64_t * bytes_written);
#endif

/*
 * Level 1 BLAS operations
 */

/**
 * `mtxdistvector_swap()' swaps values of two vectors, simultaneously
 * performing `y <- x' and `x <- y'.
 */
int mtxdistvector_swap(
    struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_copy()' copies values of a vector, `y = x'.
 */
int mtxdistvector_copy(
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_sscal()' scales a vector by a single precision
 * floating point scalar, `x = a*x'.
 */
int mtxdistvector_sscal(
    float a,
    struct mtxdistvector * x,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_dscal()' scales a vector by a double precision
 * floating point scalar, `x = a*x'.
 */
int mtxdistvector_dscal(
    double a,
    struct mtxdistvector * x,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_saxpy()' adds a vector to another vector multiplied
 * by a single precision floating point value, `y = a*x + y'.
 */
int mtxdistvector_saxpy(
    float a,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_daxpy()' adds a vector to another vector multiplied
 * by a double precision floating point value, `y = a*x + y'.
 */
int mtxdistvector_daxpy(
    double a,
    const struct mtxdistvector * x,
    struct mtxdistvector * y,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_saypx()' multiplies a vector by a single precision
 * floating point scalar and adds another vector, `y = a*y + x'.
 */
int mtxdistvector_saypx(
    float a,
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_daypx()' multiplies a vector by a double precision
 * floating point scalar and adds another vector, `y = a*y + x'.
 */
int mtxdistvector_daypx(
    double a,
    struct mtxdistvector * y,
    const struct mtxdistvector * x,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_sdot()' computes the Euclidean dot product of two
 * vectors in single precision floating point.
 */
int mtxdistvector_sdot(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float * dot,
    struct mtxmpierror * mpierror)
{
    int err;
    float dotp;
    err = mtxvector_sdot(&x->interior, &y->interior, &dotp);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        &dotp, dot, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_ddot()' computes the Euclidean dot product of two
 * vectors in double precision floating point.
 */
int mtxdistvector_ddot(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double * dot,
    struct mtxmpierror * mpierror)
{
    int err;
    double dotp;
    err = mtxvector_ddot(&x->interior, &y->interior, &dotp);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        &dotp, dot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_cdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in single
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistvector_cdotu(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float (* dot)[2],
    struct mtxmpierror * mpierror)
{
    int err;
    float dotp[2];
    err = mtxvector_cdotu(&x->interior, &y->interior, &dotp);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_zdotu()' computes the product of the transpose of a
 * complex row vector with another complex row vector in double
 * precision floating point, ‘dot := x^T*y’.
 */
int mtxdistvector_zdotu(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double (* dot)[2],
    struct mtxmpierror * mpierror)
{
    int err;
    double dotp[2];
    err = mtxvector_zdotu(&x->interior, &y->interior, &dotp);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_cdotc()' computes the Euclidean dot product of two
 * complex vectors in single precision floating point, ‘dot := x^H*y’.
 */
int mtxdistvector_cdotc(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    float (* dot)[2],
    struct mtxmpierror * mpierror)
{
    int err;
    float dotp[2];
    err = mtxvector_cdotc(&x->interior, &y->interior, &dotp);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_FLOAT, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_zdotc()' computes the Euclidean dot product of two
 * complex vectors in double precision floating point, ‘dot := x^H*y’.
 */
int mtxdistvector_zdotc(
    const struct mtxdistvector * x,
    const struct mtxdistvector * y,
    double (* dot)[2],
    struct mtxmpierror * mpierror)
{
    int err;
    double dotp[2];
    err = mtxvector_zdotc(&x->interior, &y->interior, &dotp);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        dotp, *dot, 2, MPI_DOUBLE, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_snrm2()' computes the Euclidean norm of a vector in
 * single precision floating point.
 */
int mtxdistvector_snrm2(
    const struct mtxdistvector * x,
    float * nrm2,
    struct mtxmpierror * mpierror)
{
    int err;
    float dot[2];
    err = mtxvector_cdotc(&x->interior, &x->interior, &dot);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        &dot[0], nrm2, 1, MPI_FLOAT, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrtf(*nrm2);
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_dnrm2()' computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int mtxdistvector_dnrm2(
    const struct mtxdistvector * x,
    double * nrm2,
    struct mtxmpierror * mpierror)
{
    double dot[2];
    int err = mtxvector_zdotc(&x->interior, &x->interior, &dot);
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    mpierror->mpierrcode = MPI_Allreduce(
        &dot[0], nrm2, 1, MPI_DOUBLE, MPI_SUM, x->comm);
    err = mpierror->mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
    if (mtxmpierror_allreduce(mpierror, err))
        return MTX_ERR_MPI_COLLECTIVE;
    *nrm2 = sqrt(*nrm2);
    return MTX_SUCCESS;
}

/**
 * `mtxdistvector_sasum()' computes the sum of absolute values
 * (1-norm) of a vector in single precision floating point.
 */
int mtxdistvector_sasum(
    const struct mtxdistvector * x,
    float * asum,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_dasum()' computes the sum of absolute values
 * (1-norm) of a vector in double precision floating point.
 */
int mtxdistvector_dasum(
    const struct mtxdistvector * x,
    double * asum,
    struct mtxmpierror * mpierror);

/**
 * `mtxdistvector_imax()' finds the index of the first element having
 * the maximum absolute value.
 */
int mtxdistvector_imax(
    const struct mtxdistvector * x,
    int * max,
    struct mtxmpierror * mpierror);
#endif
