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
 * Last modified: 2021-09-18
 *
 * Data types and functions for partitioning finite sets.
 */

#include <libmtx/error.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/util/partition.h>
#include <libmtx/util/index_set.h>

#include <errno.h>
#include <unistd.h>

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * `mtx_partition_type_str()' is a string representing the partition
 * type.
 */
const char * mtx_partition_type_str(
    enum mtx_partition_type type)
{
    switch (type) {
    case mtx_singleton: return "singleton";
    case mtx_block: return "block";
    case mtx_cyclic: return "cyclic";
    case mtx_block_cyclic: return "block-cyclic";
    case mtx_unstructured: return "unstructured";
    default: return mtx_strerror(MTX_ERR_INVALID_PARTITION_TYPE);
    }
}

/**
 * `mtx_parse_partition_type()' parses a string to obtain one of the
 * partition types of `enum mtx_partition_type'.
 *
 * `valid_delimiters' is either `NULL', in which case it is ignored,
 * or it is a string of characters considered to be valid delimiters
 * for the parsed string.  That is, if there are any remaining,
 * non-NULL characters after parsing, then then the next character is
 * searched for in `valid_delimiters'.  If the character is found,
 * then the parsing succeeds and the final delimiter character is
 * consumed by the parser. Otherwise, the parsing fails with an error.
 *
 * If `endptr' is not `NULL', then the address stored in `endptr'
 * points to the first character beyond the characters that were
 * consumed during parsing.
 *
 * On success, `mtx_parse_partition_type()' returns `MTX_SUCCESS' and
 * `partition_type' is set according to the parsed string and
 * `bytes_read' is set to the number of bytes that were consumed by
 * the parser.  Otherwise, an error code is returned.
 */
int mtx_parse_partition_type(
    enum mtx_partition_type * partition_type,
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
    } else if (strncmp("unstructured", t, strlen("unstructured")) == 0) {
        t += strlen("unstructured");
        *partition_type = mtx_unstructured;
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

/**
 * `mtx_partition_free()' frees resources associated with a
 * partitioning.
 */
void mtx_partition_free(
    struct mtx_partition * partition)
{
    free(partition->parts);
    for (int p = 0; p < partition->num_parts; p++)
        mtx_index_set_free(&partition->index_sets[p]);
    free(partition->index_sets);
}

/**
 * `mtx_partition_init()' initialises a partitioning of a finite set.
 */
int mtx_partition_init(
    struct mtx_partition * partition,
    enum mtx_partition_type type,
    int64_t size,
    int num_parts,
    int block_size,
    const int * parts)
{
    if (type == mtx_singleton) {
        return mtx_partition_init_singleton(partition, size);
    } else if (type == mtx_block) {
        return mtx_partition_init_block(partition, size, num_parts);
    } else if (type == mtx_cyclic) {
        return mtx_partition_init_cyclic(partition, size, num_parts);
    } else if (type == mtx_block_cyclic) {
        return mtx_partition_init_block_cyclic(
            partition, size, num_parts, block_size);
    } else if (type == mtx_unstructured) {
        return mtx_partition_init_unstructured(
            partition, size, num_parts, parts);
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
}

/**
 * `mtx_partition_init_singleton()' initialises a finite set that is
 * not partitioned.
 */
int mtx_partition_init_singleton(
    struct mtx_partition * partition,
    int64_t size)
{
    partition->type = mtx_singleton;
    partition->size = size;
    partition->num_parts = 1;
    partition->index_sets = malloc(
        partition->num_parts * sizeof(struct mtx_index_set));
    if (!partition->index_sets)
        return MTX_ERR_ERRNO;
    int err = mtx_index_set_init_interval(&partition->index_sets[0], 0, size);
    if (err) {
        free(partition->index_sets);
        return err;
    }
    partition->parts = NULL;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_init_block()' initialises a block partitioning of a
 * finite set.
 */
int mtx_partition_init_block(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    partition->type = mtx_block;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->index_sets = malloc(num_parts * sizeof(struct mtx_index_set));
    if (!partition->index_sets)
        return MTX_ERR_ERRNO;
    int64_t a = 0;
    for (int p = 0; p < num_parts; p++) {
        int64_t b = a + (size / num_parts + (p < (size % num_parts) ? 1 : 0));
        err = mtx_index_set_init_interval(&partition->index_sets[p], a, b);
        a = b;
        if (err) {
            for (int q = p-1; q > 0; q--)
                mtx_index_set_free(&partition->index_sets[q]);
            free(partition->index_sets);
            return err;
        }
    }
    partition->parts = NULL;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_init_cyclic()' initialises a cyclic partitioning of
 * a finite set.
 */
int mtx_partition_init_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts)
{
    int err;
    if (num_parts <= 0)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    partition->type = mtx_cyclic;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->index_sets = malloc(num_parts * sizeof(struct mtx_index_set));
    if (!partition->index_sets)
        return MTX_ERR_ERRNO;
    for (int p = 0; p < num_parts; p++) {
        int64_t part_size = size / num_parts + (p < (size % num_parts) ? 1 : 0);
        err = mtx_index_set_init_strided(
            &partition->index_sets[p], p, part_size, num_parts);
        if (err) {
            for (int q = p-1; q > 0; q--)
                mtx_index_set_free(&partition->index_sets[q]);
            free(partition->index_sets);
            return err;
        }
    }
    partition->parts = NULL;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_init_block_cyclic()' initialises a block-cyclic
 * partitioning of a finite set.
 */
int mtx_partition_init_block_cyclic(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts,
    int block_size);

/**
 * `mtx_partition_init_unstructured()' initialises an unstructured
 * partitioning of a finite set.
 */
int mtx_partition_init_unstructured(
    struct mtx_partition * partition,
    int64_t size,
    int num_parts,
    const int * parts)
{
    int err;
    for (int64_t i = 0; i < size; i++) {
        if (parts[i] < 0 || parts[i] >= num_parts)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    partition->type = mtx_unstructured;
    partition->size = size;
    partition->num_parts = num_parts;
    partition->index_sets = malloc(num_parts * sizeof(struct mtx_index_set));
    if (!partition->index_sets)
        return MTX_ERR_ERRNO;

    int64_t * size_per_part = malloc(num_parts * sizeof(int64_t));
    if (!size_per_part) {
        free(partition->index_sets);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < size; i++) {
        int p = parts[i];
        size_per_part[p]++;
    }

    for (int p = 0; p < num_parts; p++) {
        int64_t * indices = malloc(size_per_part[p] * sizeof(int64_t));
        if (!indices) {
            for (int q = p-1; q > 0; q--)
                mtx_index_set_free(&partition->index_sets[q]);
            free(partition->index_sets);
            free(size_per_part);
            return MTX_ERR_ERRNO;
        }

        int64_t k = 0;
        for (int64_t i = 0; i < size; i++) {
            if (parts[i] == p) {
                indices[k] = i;
                k++;
            }
        }

        err = mtx_index_set_init_discrete(
            &partition->index_sets[p], size_per_part[p], indices);
        if (err) {
            free(indices);
            for (int q = p-1; q > 0; q--)
                mtx_index_set_free(&partition->index_sets[q]);
            free(partition->index_sets);
            free(size_per_part);
            return err;
        }
        free(indices);
    }

    partition->parts = malloc(size * sizeof(int));
    if (!partition->parts) {
        for (int p = 0; p < num_parts; p++)
            mtx_index_set_free(&partition->index_sets[p]);
        free(partition->index_sets);
        free(size_per_part);
        return MTX_ERR_ERRNO;
    }
    for (int64_t i = 0; i < size; i++)
        partition->parts[i] = parts[i];

    free(size_per_part);
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_part()' determines which part of a partition that a
 * given element belongs to.
 */
int mtx_partition_part(
    const struct mtx_partition * partition,
    int * p,
    int64_t n)
{
    if (n >= partition->size)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;

    if (partition->type == mtx_singleton) {
        *p = 0;
    } else if (partition->type == mtx_block) {
        int64_t size_per_part = partition->size / partition->num_parts;
        int64_t remainder = partition->size % partition->num_parts;
        *p = n / (size_per_part+1);
        if (*p >= remainder)
            *p = remainder + (n - remainder * (size_per_part+1)) / size_per_part;
    } else if (partition->type == mtx_cyclic) {
        *p = n % partition->num_parts;
    } else if (partition->type == mtx_block_cyclic) {
        /* TODO: Not implemented. */
        errno = ENOTSUP;
        return MTX_ERR_ERRNO;
    } else if (partition->type == mtx_unstructured) {
        *p = partition->parts[n];
    } else {
        return MTX_ERR_INVALID_PARTITION_TYPE;
    }
    return MTX_SUCCESS;
}

/*
 * I/O functions
 */

/**
 * `mtx_partition_read_parts()' reads the part numbers assigned to
 * each element of a partitioned set from the given path.  The path
 * must be to a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If `path' is `-', then standard input is used.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_partition_read_parts(
    struct mtx_partition * partition,
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
    err = mtx_partition_fread_parts(
        partition, num_parts, f, lines_read, bytes_read, 0, NULL);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_fread_parts()' reads the part numbers assigned to
 * each element of a partitioned set from a stream formatted as a
 * Matrix Market file.  The Matrix Market file must be in the form of
 * an integer vector in array format.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_partition_fread_parts(
    struct mtx_partition * partition,
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

    err = mtx_partition_init(
        partition, mtx_unstructured, mtxfile.size.num_rows, num_parts, 0,
        mtxfile.data.array_integer_single);
    if (err) {
        mtxfile_free(&mtxfile);
        return err;
    }
    mtxfile_free(&mtxfile);
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_fread_indices()' reads the global indices of
 * elements belonging to a given part of a partitioned set from a
 * stream formatted as a Matrix Market file.  The Matrix Market file
 * must be in the form of an integer vector in array format.
 *
 * If an error code is returned, then `lines_read' and `bytes_read'
 * are used to return the line number and byte at which the error was
 * encountered during the parsing of the Matrix Market file.
 */
int mtx_partition_fread_indices(
    struct mtx_partition * partition,
    int part,
    FILE * f,
    int * lines_read,
    int64_t * bytes_read,
    size_t line_max,
    char * linebuf);

/**
 * `mtx_partition_write_parts()' writes the part numbers assigned to
 * each element of a partitioned set to the given path.  The file is
 * written as a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 */
int mtx_partition_write_parts(
    const struct mtx_partition * partition,
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
    err = mtx_partition_fwrite_parts(partition, f, format, bytes_written);
    if (err) {
        fclose(f);
        return err;
    }
    fclose(f);
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_fwrite_parts()' writes the part numbers assigned to
 * each element of a partitioned set to a stream formatted as a Matrix
 * Market file.  The Matrix Market file is written in the form of an
 * integer vector in array format.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_partition_fwrite_parts(
    const struct mtx_partition * partition,
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
    } else if (partition->type == mtx_unstructured) {
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

/**
 * `mtx_partition_write_permutation()' writes the permutation of a
 * given part of a partitioned set to the given path.  The permutation
 * is represented by an array of global indices of the elements
 * belonging to the given part prior to partitioning.  The file is
 * written as a Matrix Market file in the form of an integer vector in
 * array format.
 *
 * If `path' is `-', then standard output is used.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_partition_write_permutation(
    const struct mtx_partition * partition,
    int part,
    const char * path,
    const char * format,
    int64_t * bytes_written)
{
    if (part < 0 || part >= partition->num_parts)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    return mtx_index_set_write(
        &partition->index_sets[part], path, format, bytes_written);
}

/**
 * `num_places()` is the number of digits or places in a given
 * non-negative integer.
 */
static int num_places(int n, int * places)
{
    int r = 1;
    if (n < 0)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    while (n > 9) {
        n /= 10;
        r++;
    }
    *places = r;
    return 0;
}

/**
 * `format_path()` formats a path by replacing any occurence of '%p'
 * in `pathfmt' with the number `part'.
 */
static int format_path(
    const char * pathfmt,
    char ** output_path,
    int part,
    int comm_size)
{
    int err;
    int part_places;
    err = num_places(comm_size, &part_places);
    if (err)
        return err;

    /* Count the number of occurences of '%p' in the format string. */
    int count = 0;
    const char * needle = "%p";
    const int needle_len = strlen(needle);
    const int pathfmt_len = strlen(pathfmt);

    const char * src = pathfmt;
    const char * next;
    while ((next = strstr(src, needle))) {
        count++;
        src = next + needle_len;
        assert(src < pathfmt + pathfmt_len);
    }
    if (count < 1)
        return MTX_ERR_INVALID_PATH_FORMAT;

    /* Allocate storage for the path. */
    int path_len = pathfmt_len + (part_places-needle_len)*count;
    char * path = malloc(path_len+1);
    if (!path)
        return errno;
    path[path_len] = '\0';

    src = pathfmt;
    char * dest = path;
    while ((next = strstr(src, needle))) {
        /* Copy the format string up until the needle, '%p'. */
        while (src < next && dest <= path + path_len)
            *dest++ = *src++;
        src += needle_len;

        /* Replace '%p' with the number of the current part. */
        assert(dest + part_places <= path + path_len);
        int len = snprintf(dest, part_places+1, "%0*d", part_places, part);
        assert(len == part_places);
        dest += part_places;
    }

    /* Copy the remainder of the format string. */
    while (*src != '\0' && dest <= path + path_len)
        *dest++ = *src++;
    assert(dest == path + path_len);
    *dest = '\0';

    *output_path = path;
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_write_permutations()' writes the permutations for
 * each part of a partitioned set to the given path.  The permutation
 * is represented by an array of global indices of the elements
 * belonging each part prior to partitioning.  The file for each part
 * is written as a Matrix Market file in the form of an integer vector
 * in array format.
 *
 * Each occurrence of '%p' in `pathfmt' is replaced by the number of
 * each part number.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_partition_write_permutations(
    const struct mtx_partition * partition,
    const char * pathfmt,
    const char * format,
    int64_t * bytes_written)
{
    int err;
    for (int p = 0; p < partition->num_parts; p++) {
        char * path;
        err = format_path(pathfmt, &path, p, partition->num_parts);
        if (err)
            return err;
        err = mtx_partition_write_permutation(
            partition, p, path, format, bytes_written);
        if (err) {
            free(path);
            return err;
        }
        free(path);
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_partition_write_permutation()' writes the permutation of a
 * given part of a partitioned set to a stream as a Matrix Market
 * file.  The permutation is represented by an array of global indices
 * of the elements belonging to the given part prior to partitioning.
 * The file is written as a Matrix Market file in the form of an
 * integer vector in array format.
 *
 * If `format' is not `NULL', then the given format string is used
 * when printing numerical values.  The format specifier must be '%d',
 * and a fixed field width may optionally be specified (e.g., "%3d"),
 * but variable field width (e.g., "%*d"), as well as length modifiers
 * (e.g., "%ld") are not allowed.  If `format' is `NULL', then the
 * format specifier '%d' is used.
 *
 * If it is not `NULL', then the number of bytes written to the stream
 * is returned in `bytes_written'.
 */
int mtx_partition_fwrite_permutation(
    const struct mtx_partition * partition,
    int part,
    FILE * f,
    const char * format,
    int64_t * bytes_written)
{
    if (part < 0 || part >= partition->num_parts)
        return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    return mtx_index_set_fwrite(
        &partition->index_sets[part], f, format, bytes_written);
}
