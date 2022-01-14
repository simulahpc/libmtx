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
 * Last modified: 2022-01-11
 *
 * Data types and functions for partitioning finite sets.
 */

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/index_set.h>

#include <errno.h>
#include <unistd.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Types of partitioning
 */

/**
 * ‘mtxpartitioning_str()’ is a string representing the partition
 * type.
 */
const char * mtxpartitioning_str(
    enum mtxpartitioning type)
{
    switch (type) {
    case mtx_singleton: return "singleton";
    case mtx_block: return "block";
    case mtx_cyclic: return "cyclic";
    case mtx_block_cyclic: return "block-cyclic";
    case mtx_custom_partition: return "custom";
    default: return mtxstrerror(MTX_ERR_INVALID_PARTITION_TYPE);
    }
}

/**
 * ‘mtxpartitioning_parse()’ parses a string to obtain one of the
 * partition types of ‘enum mtxpartitioning’.
 *
 * ‘valid_delimiters’ is either ‘NULL’, in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in ‘valid_delimiters’.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If ‘endptr’ is not ‘NULL’, then the address stored in ‘endptr’
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, ‘mtxpartitioning_parse()’ returns ‘MTX_SUCCESS’ and
 * ‘partition_type’ is set according to the parsed string and
 * ‘bytes_read’ is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtxpartitioning_parse(
    enum mtxpartitioning * partition_type,
    int64_t * bytes_read,
    const char ** endptr,
    const char * s,
    const char * valid_delimiters)
{
    const char * t = s;
    if (strncmp("singleton", t, strlen("singleton")) == 0) {
        t += strlen("singleton");
        *partition_type = mtx_singleton;
    } else if (strncmp("block", t, strlen("block")) == 0) {
        t += strlen("block");
        *partition_type = mtx_block;
    } else if (strncmp("cyclic", t, strlen("cyclic")) == 0) {
        t += strlen("cyclic");
        *partition_type = mtx_cyclic;
    } else if (strncmp("block-cyclic", t, strlen("block-cyclic")) == 0) {
        t += strlen("block-cyclic");
        *partition_type = mtx_block_cyclic;
    } else if (strncmp("custom", t, strlen("custom")) == 0) {
        t += strlen("custom");
        *partition_type = mtx_custom_partition;
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    if (valid_delimiters && *t != '\0') {
        if (!strchr(valid_delimiters, *t))
            return MTX_ERR_INVALID_PARTITION_TYPE;
        t++;
    }
    if (bytes_read)
        *bytes_read += t-s;
    if (endptr)
        *endptr = t;
    return MTX_SUCCESS;
}

/*
 * Partitions of finite sets
 */

/**
 * ‘mtxpartition_free()’ frees resources associated with a
 * partitioning.
 */
void mtxpartition_free(
    struct mtxpartition * partition)
{
    free(partition->elements_per_part);
    free(partition->parts);
    free(partition->parts_ptr);
    free(partition->part_sizes);
}

/**
 * ‘mtxpartition_init()’ initialises a partitioning of a finite set.
 */
int mtxpartition_init(
    struct mtxpartition * partition,
    enum mtxpartitioning type,
    int64_t size,
    int num_parts,
    const int64_t * part_sizes,
    int block_size,
    const int * parts)
{
    if (type == mtx_singleton && num_parts == 1) {
        return mtxpartition_init_singleton(partition, size);
    } else if (type == mtx_block) {
        return mtxpartition_init_block(partition, size, num_parts, part_sizes);
    } else if (type == mtx_cyclic) {
        return mtxpartition_init_cyclic(partition, size, num_parts);
    } else if (type == mtx_block_cyclic) {
        return mtxpartition_init_block_cyclic(
            partition, size, num_parts, block_size);
    } else if (type == mtx_custom_partition) {
        return mtxpartition_init_custom(
            partition, size, num_parts, parts);
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
}

/**
 * ‘mtxpartition_copy()’ creates a copy of a partitioning.
 */
int mtxpartition_copy(
    struct mtxpartition * dst,
    const struct mtxpartition * src)
{
    dst->type = src->type;
    dst->size = src->size;
    dst->num_parts = src->num_parts;
    dst->part_sizes = malloc(dst->num_parts * sizeof(int64_t));
    if (!dst->part_sizes)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < dst->num_parts; p++)
        dst->part_sizes[p] = src->part_sizes[p];
    dst->parts_ptr = malloc((dst->num_parts+1) * sizeof(int64_t));
    if (!dst->parts_ptr) {
        free(dst->part_sizes);
        return MTX_ERR_ERRNO;
    }
    for (int p = 0; p <= dst->num_parts; p++)
        dst->parts_ptr[p] = src->parts_ptr[p];
    if (src->parts) {
        dst->parts = malloc(dst->size * sizeof(int64_t));
        if (!dst->parts) {
            free(dst->parts_ptr);
            free(dst->part_sizes);
            return MTX_ERR_ERRNO;
        }
        for (int64_t i = 0; i < dst->size; i++)
            dst->parts[i] = src->parts[i];
    } else {
        dst->parts = NULL;
    }
    if (src->elements_per_part) {
        dst->elements_per_part = malloc(dst->size * sizeof(int64_t));
        if (!dst->elements_per_part) {
            free(dst->parts);
            free(dst->parts_ptr);
            free(dst->part_sizes);
            return MTX_ERR_ERRNO;
        }
        for (int64_t i = 0; i < dst->size; i++)
            dst->elements_per_part[i] = src->elements_per_part[i];
    } else {
        dst->elements_per_part = NULL;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_singleton()’ initialises a finite set that is
 * not partitioned.
 */
int mtxpartition_init_singleton(
    struct mtxpartition * partition,
    int64_t size)
{
    if (size > INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }
    partition->type = mtx_singleton;
    partition->size = size;
    partition->num_parts = 1;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;
    partition->part_sizes[0] = size;
    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    partition->parts_ptr[1] = size;
    partition->parts = NULL;
    partition->elements_per_part = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_block()’ initialises a block partitioning of a
 * finite set. Each block is made up of a contiguous set of elements,
 * but blocks may vary in size.
 *
 * If ‘part_sizes’ is ‘NULL’, then the elements are divided into
 * blocks of equal size. Otherwise, ‘part_sizes’ must point to an
 * array of length ‘num_parts’ containing the number of elements in
 * each part. Moreover, the sum of the entries in ‘part_sizes’ must be
 * equal to ‘size’.
 */
int mtxpartition_init_block(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int64_t * part_sizes)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (size / num_parts >= INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }
    partition->type = mtx_block;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;
    if (part_sizes) {
        for (int p = 0; p < num_parts; p++)
            partition->part_sizes[p] = part_sizes[p];
    } else {
        for (int p = 0; p < num_parts; p++) {
            partition->part_sizes[p] =
                (size / num_parts + (p < (size % num_parts) ? 1 : 0));
        }
    }

    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }
    if (partition->parts_ptr[num_parts] != size) {
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    partition->parts = NULL;
    partition->elements_per_part = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_cyclic()’ initialises a cyclic partitioning of
 * a finite set.
 */
int mtxpartition_init_cyclic(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (size / num_parts >= INT_MAX) {
        errno = ERANGE;
        return MTX_ERR_ERRNO;
    }

    partition->type = mtx_cyclic;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        partition->part_sizes[p] =
            (size / num_parts + (p < (size % num_parts) ? 1 : 0));
    }

    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }
    partition->parts = NULL;
    partition->elements_per_part = NULL;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_init_block_cyclic()’ initialises a block-cyclic
 * partitioning of a finite set.
 */
int mtxpartition_init_block_cyclic(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    int block_size)
{
    errno = ENOTSUP;
    return MTX_ERR_ERRNO;
}

/**
 * ‘mtxpartition_init_custom()’ initialises a user-defined
 * partitioning of a finite set.
 */
int mtxpartition_init_custom(
    struct mtxpartition * partition,
    int64_t size,
    int num_parts,
    const int * parts)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    for (int64_t i = 0; i < size; i++) {
        if (parts[i] < 0 || parts[i] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }

    partition->type = mtx_custom_partition;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->part_sizes = malloc(partition->num_parts * sizeof(int64_t));
    if (!partition->part_sizes)
        return MTX_ERR_ERRNO;

    for (int p = 0; p < num_parts; p++)
        partition->part_sizes[p] = 0;
    for (int64_t k = 0; k < size; k++)
        partition->part_sizes[parts[k]]++;

    partition->parts_ptr = malloc((partition->num_parts+1) * sizeof(int64_t));
    if (!partition->parts_ptr) {
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }

    partition->parts = malloc(partition->size * sizeof(int64_t));
    if (!partition->parts) {
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < size; i++)
        partition->parts[i] = parts[i];

    partition->elements_per_part = malloc(partition->size * sizeof(int64_t));
    if (!partition->elements_per_part) {
        free(partition->parts);
        free(partition->parts_ptr);
        free(partition->part_sizes);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < size; i++) {
        int p = parts[i];
        partition->elements_per_part[
            partition->parts_ptr[p]] = i;
        partition->parts_ptr[p]++;
    }
    partition->parts_ptr[0] = 0;
    for (int p = 0; p < num_parts; p++) {
        partition->parts_ptr[p+1] =
            partition->parts_ptr[p] + partition->part_sizes[p];
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_assign()’ assigns part numbers to elements of an
 * array according to the partitioning.
 *
 * The arrays ‘elements’ and ‘parts’ must both contain enough storage
 * for ‘size’ values of type ‘int’. If successful, ‘parts’ will
 * contain the part numbers of each element in the ‘elements’ array.
 */
int mtxpartition_assign(
    const struct mtxpartition * partition,
    int64_t size,
    const int64_t * elements,
    int * parts)
{
    if (partition->type == mtx_singleton) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            parts[k] = 0;
        }

    } else if (partition->type == mtx_block) {
        int64_t size_per_part = partition->size / partition->num_parts;
        int64_t remainder = partition->size % partition->num_parts;
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            int64_t n = elements[k];
            parts[k] = n / (size_per_part+1);
            if (parts[k] >= remainder)
                parts[k] = remainder +
                    (n - remainder * (size_per_part+1)) / size_per_part;
        }

    } else if (partition->type == mtx_cyclic) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            parts[k] = elements[k] % partition->num_parts;
        }

    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (partition->type == mtx_custom_partition) {
        for (int64_t k = 0; k < size; k++) {
            if (elements[k] < 0 || elements[k] >= partition->size)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            parts[k] = partition->parts[elements[k]];
        }

    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_globalidx()’ translates from a local numbering of
 * elements within a given part to a global numbering of elements in
 * the partitioned set.
 *
 * The argument ‘part’ denotes the part of the partition for which the
 * local element numbers are given.
 *
 * The arrays ‘localelem’ and ‘globalelem’ must be of length equal to
 * ‘size’. The former is used to specify the local element numbers
 * within the specified part, and must therefore contain values in the
 * range ‘0, 1, ..., partition->part_sizes[part]-1’. If successful,
 * the array ‘globalelem’ will contain the global numbers
 * corresponding to each of the local element numbers in ‘localelem’.
 *
 * If needed, ‘localelem’ and ‘globalelem’ are allowed to point to the
 * same underlying array. The values of ‘localelem’ will then be
 * overwritten by the global element numbers.
 */
int mtxpartition_globalidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int64_t * localelem,
    int64_t * globalelem)
{
    if (part < 0 || part >= partition->num_parts)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (partition->type == mtx_singleton) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            globalelem[k] = localelem[k];
        }

    } else if (partition->type == mtx_block) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            globalelem[k] = partition->parts_ptr[part] + localelem[k];
        }

    } else if (partition->type == mtx_cyclic) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            globalelem[k] = part + partition->num_parts * localelem[k];
        }

    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (partition->type == mtx_custom_partition) {
        for (int64_t k = 0; k < size; k++) {
            if (localelem[k] < 0 || localelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            int64_t offset = partition->parts_ptr[part];
            globalelem[k] = partition->elements_per_part[offset + localelem[k]];
        }

    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_localidx()’ translates from a global numbering of
 * elements in the partitioned set to a local numbering of elements
 * within a given part.
 *
 * The argument ‘part’ denotes the part of the partition for which the
 * local element numbers are obtained.
 *
 * The arrays ‘globalelem’ and ‘localelem’ must be of length equal to
 * ‘size’. The former is used to specify the global element numbers of
 * elements belonging to the specified part. 
 *
 * If successful, the array ‘localelem’ will contain local element
 * numbers in the range ‘0, 1, ..., partition->part_sizes[part]-1’
 * that were obtained by translating from the global element numbers
 * in ‘globalelem’.
 *
 * If needed, ‘globalelem’ and ‘localelem’ are allowed to point to the
 * same underlying array. The values of ‘globalelem’ will then be
 * overwritten by the local element numbers.
 */
int mtxpartition_localidx(
    const struct mtxpartition * partition,
    int part,
    int64_t size,
    const int64_t * globalelem,
    int64_t * localelem)
{
    if (part < 0 || part >= partition->num_parts)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (partition->type == mtx_singleton) {
        for (int64_t k = 0; k < size; k++) {
            if (globalelem[k] < 0 || globalelem[k] >= partition->part_sizes[part])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            localelem[k] = globalelem[k];
        }

    } else if (partition->type == mtx_block) {
        for (int64_t k = 0; k < size; k++) {
            if (globalelem[k] < partition->parts_ptr[part] ||
                globalelem[k] >= partition->parts_ptr[part+1])
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            localelem[k] = globalelem[k] - partition->parts_ptr[part];
        }

    } else if (partition->type == mtx_cyclic) {
        for (int64_t k = 0; k < size; k++) {
            if (globalelem[k] % partition->num_parts != part)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
            localelem[k] = (globalelem[k] - part) / partition->num_parts;
        }

    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;

    } else if (partition->type == mtx_custom_partition) {
        for (int64_t k = 0; k < size; k++) {
            bool found = false;
            for (int64_t l = 0; l < partition->part_sizes[part]; l++) {
                if (partition->elements_per_part[l] == globalelem[k]) {
                    localelem[k] = l;
                    found = true;
                    break;
                }
            }
            if (!found)
                return MTX_ERR_INDEX_OUT_OF_BOUNDS;
        }

    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * ‘mtxpartition_read_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from the given path.  The path
 * must be to a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_read_parts(
    struct mtxpartition * partition,
    int num_parts,
    const char * path,
    int * lines_read,
    int64_t * bytes_read)
{
    int err;
    *lines_read = -1;
    *bytes_read = 0;

    FILE * f;
    if (strcmp(path, "-") == 0) {
        int fd = dup(STDIN_FILENO);
        if (fd == -1)
            return MTX_ERR_ERRNO;
        if ((f = fdopen(fd, "r")) == NULL) {
            close(fd);
            return MTX_ERR_ERRNO;
        }
    } else if ((f = fopen(path, "r")) == NULL) {
        return MTX_ERR_ERRNO;
    }
    *lines_read = 0;
    err = mtxpartition_fread_parts(
        partition, num_parts, f, lines_read, bytes_read, 0, NULL);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_fread_parts()’ reads the part numbers assigned to
 * each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_fread_parts(
    struct mtxpartition * partition,
    int num_parts,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_fread(
        &mtxfile, mtx_single, f, lines_read, bytes_read, line_max, linebuf);
    if (err)
        return err;

    if (mtxfile.header.object != mtxfile_vector) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_MTX_OBJECT;
    } else if (mtxfile.header.format != mtxfile_array) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_MTX_FORMAT;
    } else if (mtxfile.header.field != mtxfile_integer) {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    err = mtxpartition_init_custom(
        partition, mtxfile.size.num_rows, num_parts,
        mtxfile.data.array_integer_single);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_fread_indices()’ reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then ‘lines_read’ and ‘bytes_read’
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtxpartition_fread_indices(
    struct mtxpartition * partition,
    int part,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * ‘mtxpartition_write_parts()’ writes the part numbers assigned to
 * each element of a partitioned set to the given path.  The file is
 * written as a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If ‘path’ is ‘-’, then standard output is used.
 *
 * If ‘format’ is not ‘NULL’, then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If ‘format’ is ‘NULL’, then the
 * format specifier '%d' is used.
 */
int mtxpartition_write_parts(
    const struct mtxpartition * partition,
    const char * path,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    *bytes_written = 0;

    FILE * f;
    if (strcmp(path, "-") == 0) {
        int fd = dup(STDOUT_FILENO);
        if (fd == -1)
            return MTX_ERR_ERRNO;
        if ((f = fdopen(fd, "w")) == NULL) {
            close(fd);
            return MTX_ERR_ERRNO;
        }
    } else if ((f = fopen(path, "w")) == NULL) {
        return MTX_ERR_ERRNO;
    }
    err = mtxpartition_fwrite_parts(partition, f, format, bytes_written);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * ‘mtxpartition_fwrite_parts()’ writes the part numbers assigned to
 * each element of a partitioned set to a stream formatted as a Matrix
 * Market file.  The Matrix Market file is written in the form of an
 * integer vector in array format.
 *
 * If ‘format’ is not ‘NULL’, then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If ‘format’ is ‘NULL’, then the
 * format specifier '%d' is used.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 */
int mtxpartition_fwrite_parts(
    const struct mtxpartition * partition,
    FILE * f,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    struct mtxfile mtxfile;
    err = mtxfile_alloc_vector_array(
        &mtxfile, mtxfile_integer, mtx_single, partition->size);
    if (err)
        return err;

    int * parts = mtxfile.data.array_integer_single;
    if (partition->type == mtx_singleton) {
        for (int64_t i = 0; i < partition->size; i++)
            parts[i] = 0;
    } else if (partition->type == mtx_block) {
        int64_t size_per_part = partition->size / partition->num_parts;
        int64_t remainder = partition->size % partition->num_parts;
        for (int64_t i = 0; i < partition->size; i++) {
            parts[i] = i / (size_per_part+1);
            if (parts[i] >= remainder) {
                parts[i] = remainder +
                    (i - remainder * (size_per_part+1)) / size_per_part;
            }
        }
    } else if (partition->type == mtx_cyclic) {
        for (int64_t i = 0; i < partition->size; i++)
            parts[i] = i % partition->num_parts;
    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        mtxfile_free(&mtxfile);
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (partition->type == mtx_custom_partition) {
        for (int64_t i = 0; i < partition->size; i++)
            parts[i] = partition->parts[i];
    } else {
        mtxfile_free(&mtxfile);
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }

    err = mtxfile_fwrite(&mtxfile, f, format, bytes_written);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}
