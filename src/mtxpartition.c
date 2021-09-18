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
 * Last modified: 2021-09-10
 *
 * Partition a Matrix Market file and write parts to separate files.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#include <errno.h>

#include <assert.h>
#include <inttypes.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxpartition";
const char * program_version = "0.1.0";
const char * program_copyright =
    "Copyright (C) 2021 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

/**
 * `program_options` contains data to related program options.
 */
struct program_options
{
    char * mtx_path;
    enum mtx_precision precision;
    bool gzip;
    char * mtx_output_path;
    char * format;
    int num_row_parts;
    enum mtx_partition_type row_partition;
    char * row_partition_path;
    char * partition_output_path;
    int verbose;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->mtx_path = NULL;
    args->precision = mtx_double;
    args->gzip = false;
    args->mtx_output_path = strdup("out%p.mtx");
    args->partition_output_path = NULL;
    args->format = NULL;
    args->num_row_parts = 1;
    args->row_partition = mtx_block;
    args->row_partition_path = NULL;
    args->verbose = 0;
    return 0;
}

/**
 * `program_options_free()` frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->mtx_path)
        free(args->mtx_path);
    if (args->mtx_output_path)
        free(args->mtx_output_path);
    if (args->partition_output_path)
        free(args->partition_output_path);
    if (args->row_partition_path)
        free(args->row_partition_path);
    if (args->format)
        free(args->format);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Partition a Matrix Market file and write parts to separate files.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: single or double. (default: double)\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --output-path=FILE\tpath for partitioned Matrix Market files, where\n");
    fprintf(f, "\t\t\t'%%p' in the string is replaced with the number of\n");
    fprintf(f, "\t\t\tof each part (default: out%%p.mtx).\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  --partition-output-path=FILE\tpath to Matrix Market file for outputting\n");
    fprintf(f, "\t\t\t\tpart numbers assigned to each matrix or vector entry.\n");
    fprintf(f, "  --row-parts=N\t\tnumber of parts to use when partitioning rows.\n");
    fprintf(f, "  --row-partition=TYPE\tmethod of partitioning matrix or vector rows:\n");
    fprintf(f, "\t\t\tblock, cyclic, block-cyclic, unstructured. (default: block)\n");
    fprintf(f, "  --row-partition-path=FILE\t"
            "path to Matrix Market file with a row partition\n");
    fprintf(f, "\t\t\t\tto use when the row partition is `unstructured'.\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help\t\tdisplay this help and exit\n");
    fprintf(f, "  --version\t\tdisplay version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * `program_options_print_version()` prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s (libmtx %s)\n", program_name, program_version,
            libmtx_version);
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * `parse_program_options()` parses program options.
 */
static int parse_program_options(
    int * argc,
    char *** argv,
    struct program_options * args)
{
    int err;

    /* Set program invocation name. */
    program_invocation_name = (*argv)[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);
    (*argc)--; (*argv)++;

    /* Set default program options. */
    err = program_options_init(args);
    if (err)
        return err;

    /* Parse program options. */
    int num_arguments_consumed = 0;
    while (*argc > 0) {
        *argc -= num_arguments_consumed;
        *argv += num_arguments_consumed;
        num_arguments_consumed = 0;
        if (*argc <= 0)
            break;

        if (strcmp((*argv)[0], "--precision") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            if (strcmp(s, "single") == 0) {
                args->precision = mtx_single;
            } else if (strcmp(s, "double") == 0) {
                args->precision = mtx_double;
            } else {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--precision=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--precision=");
            if (strcmp(s, "single") == 0) {
                args->precision = mtx_single;
            } else if (strcmp(s, "double") == 0) {
                args->precision = mtx_double;
            } else {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "-z") == 0 ||
            strcmp((*argv)[0], "--gzip") == 0 ||
            strcmp((*argv)[0], "--gunzip") == 0 ||
            strcmp((*argv)[0], "--ungzip") == 0)
        {
            args->gzip = true;
            num_arguments_consumed++;
            continue;
        }

        /* Parse output path. */
        if (strcmp((*argv)[0], "--output-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->mtx_output_path)
                free(args->mtx_output_path);
            args->mtx_output_path = strdup((*argv)[1]);
            if (!args->mtx_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--output-path=") == (*argv)[0]) {
            if (args->mtx_output_path)
                free(args->mtx_output_path);
            args->mtx_output_path =
                strdup((*argv)[0] + strlen("--output-path="));
            if (!args->mtx_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--partition-output-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->partition_output_path)
                free(args->partition_output_path);
            args->partition_output_path = strdup((*argv)[1]);
            if (!args->partition_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--partition-output-path=") == (*argv)[0]) {
            if (args->partition_output_path)
                free(args->partition_output_path);
            args->partition_output_path =
                strdup((*argv)[0] + strlen("--partition-output-path="));
            if (!args->partition_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--format") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            args->format = strdup((*argv)[1]);
            if (!args->format) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--format=") == (*argv)[0]) {
            args->format = strdup((*argv)[0] + strlen("--format="));
            if (!args->format) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Parse partitioning options. */
        if (strcmp((*argv)[0], "--row-partition") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtx_parse_partition_type(
                &args->row_partition, NULL, NULL, s, NULL);
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--row-partition=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--row-partition=");
            err = mtx_parse_partition_type(
                &args->row_partition, NULL, NULL, s, NULL);
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--row-parts") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->num_row_parts, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--row-parts=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--row-parts="), NULL,
                &args->num_row_parts, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--row-partition-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->row_partition_path)
                free(args->row_partition_path);
            args->row_partition_path = strdup((*argv)[1]);
            if (!args->row_partition_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--row-partition-path=") == (*argv)[0]) {
            if (args->row_partition_path)
                free(args->row_partition_path);
            args->row_partition_path =
                strdup((*argv)[0] + strlen("--row-partition-path="));
            if (!args->row_partition_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "-v") == 0 || strcmp((*argv)[0], "--verbose") == 0) {
            args->verbose++;
            num_arguments_consumed++;
            continue;
        }

        /* If requested, print program help text. */
        if (strcmp((*argv)[0], "-h") == 0 || strcmp((*argv)[0], "--help") == 0) {
            program_options_free(args);
            program_options_print_help(stdout);
            exit(EXIT_SUCCESS);
        }

        /* If requested, print program version information. */
        if (strcmp((*argv)[0], "--version") == 0) {
            program_options_free(args);
            program_options_print_version(stdout);
            exit(EXIT_SUCCESS);
        }

        /* Stop parsing options after '--'.  */
        if (strcmp((*argv)[0], "--") == 0) {
            (*argc)--; (*argv)++;
            break;
        }

        /* Parse Matrix Market input path ('-' is used for stdin). */
        if ((strlen((*argv)[0]) > 0 && (*argv)[0][0] != '-') ||
            (strlen((*argv)[0]) == 1 && (*argv)[0][0] == '-'))
        {
            if (args->mtx_path)
                free(args->mtx_path);
            args->mtx_path = strdup((*argv)[0]);
            if (!args->mtx_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Unrecognised option. */
        program_options_free(args);
        return EINVAL;
    }

    return 0;
}

/**
 * `timespec_duration()` is the duration, in seconds, elapsed between
 * two given time points.
 */
static double timespec_duration(
    struct timespec t0,
    struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/**
 * `num_places()` is the number of digits or places in a given
 * non-negative integer.
 */
static int num_places(int n, int * places)
{
    int r = 1;
    if (n < 0)
        return EINVAL;
    while (n > 9) {
        n /= 10;
        r++;
    }
    *places = r;
    return 0;
}

/**
 * `format_path()` formats a path by replacing any occurence of '%p'
 * in `path_fmt' with the number `part'.
 */
static int format_path(
    const char * path_fmt,
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
    const int path_fmt_len = strlen(path_fmt);

    const char * src = path_fmt;
    const char * next;
    while ((next = strstr(src, needle))) {
        count++;
        src = next + needle_len;
        assert(src < path_fmt + path_fmt_len);
    }

    /* Allocate storage for the path. */
    int path_len = path_fmt_len + (part_places-needle_len)*count;
    char * path = malloc(path_len+1);
    if (!path)
        return errno;
    path[path_len] = '\0';

    src = path_fmt;
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
    return 0;
}

/**
 * `main()`.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    FILE * diagf = stderr;
    setlocale(LC_ALL, "");

    /* 1. Parse program options. */
    struct program_options args;
    int argc_copy = argc;
    char ** argv_copy = argv;
    err = parse_program_options(&argc_copy, &argv_copy, &args);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv_copy[0]);
        return EXIT_FAILURE;
    }
    if (!args.mtx_path) {
        fprintf(stderr, "%s: Please specify a Matrix Market file\n",
                program_invocation_short_name);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (args.row_partition == mtx_unstructured && !args.row_partition_path) {
        fprintf(stderr, "%s: Please specify a Matrix Market file "
                "with --row-partition-path\n",
                program_invocation_short_name);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 2. Read a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    int lines_read;
    int64_t bytes_read;
    setlocale(LC_ALL, "C");
    err = mtxfile_read(
        &mtxfile, args.precision, args.mtx_path, args.gzip,
        &lines_read, &bytes_read);
    setlocale(LC_ALL, "");
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d: %s\n",
                program_invocation_short_name,
                args.mtx_path, lines_read+1,
                mtx_strerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.mtx_path, mtx_strerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    struct mtxfile row_partition_mtxfile;
    int * rowparts = NULL;
    if (args.row_partition == mtx_unstructured && args.row_partition_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read;
        int64_t bytes_read;
        setlocale(LC_ALL, "C");
        err = mtxfile_read(
            &row_partition_mtxfile, mtx_single, args.row_partition_path, args.gzip,
            &lines_read, &bytes_read);
        setlocale(LC_ALL, "");
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.row_partition_path, lines_read+1,
                    mtx_strerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.row_partition_path, mtx_strerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
        }

        if (row_partition_mtxfile.header.object != mtxfile_vector ||
            row_partition_mtxfile.header.format != mtxfile_array ||
            row_partition_mtxfile.header.field != mtxfile_integer ||
            row_partition_mtxfile.size.num_rows != mtxfile.size.num_rows)
        {
            fprintf(stderr, "%s: %s: incompatible Matrix Market file - "
                    "expected integer vector in array format with %d rows\n",
                    program_invocation_short_name,
                    args.row_partition_path, mtxfile.size.num_rows);
            mtxfile_free(&row_partition_mtxfile);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        rowparts = row_partition_mtxfile.data.array_integer_single;
    }

    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_partition_rows: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* 3. Partition the rows of the matrix or vector. */
    struct mtx_partition row_partition;
    err = mtx_partition_init(
        &row_partition, args.row_partition,
        mtxfile.size.num_rows, args.num_row_parts, 0, rowparts);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, mtx_strerror(err));
        if (args.row_partition == mtx_unstructured && args.row_partition_path)
            mtxfile_free(&row_partition_mtxfile);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (args.row_partition == mtx_unstructured && args.row_partition_path)
        mtxfile_free(&row_partition_mtxfile);

    int64_t * data_lines_per_part_ptr =
        malloc((args.num_row_parts+1) * sizeof(int64_t));
    if (!data_lines_per_part_ptr) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        mtx_partition_free(&row_partition);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t num_data_lines;
    err = mtxfile_size_num_data_lines(&mtxfile.size, &num_data_lines);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, mtx_strerror(err));
        free(data_lines_per_part_ptr);
        mtx_partition_free(&row_partition);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int * part_per_data_line = malloc(num_data_lines * sizeof(int));
    if (!part_per_data_line) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(data_lines_per_part_ptr);
        mtx_partition_free(&row_partition);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    err = mtxfile_partition_rows(
        &mtxfile, &row_partition, data_lines_per_part_ptr, part_per_data_line);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, mtx_strerror(err));
        free(part_per_data_line);
        free(data_lines_per_part_ptr);
        mtx_partition_free(&row_partition);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    /* 4. Write a Matrix Market file for each part. */
    if (args.mtx_output_path) {
        for (int p = 0; p < args.num_row_parts; p++) {
            /* Extract a Matrix Market file for the current
             * partition. */
            struct mtxfile mtxfile_p;
            err = mtxfile_init_from_row_partition(
                &mtxfile_p, &mtxfile, &row_partition, data_lines_per_part_ptr, p);
            if (err) {
                if (args.verbose > 0)
                    fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtx_strerror(err));
                free(part_per_data_line);
                free(data_lines_per_part_ptr);
                mtx_partition_free(&row_partition);
                mtxfile_free(&mtxfile);
                program_options_free(&args);
                return EXIT_FAILURE;
            }

            err = mtxfile_comments_printf(
                &mtxfile_p.comments, "%% This file was generated by %s %s\n",
                program_name, program_version);
            if (err) {
                if (args.verbose > 0)
                    fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtx_strerror(err));
                mtxfile_free(&mtxfile_p);
                free(part_per_data_line);
                free(data_lines_per_part_ptr);
                mtx_partition_free(&row_partition);
                mtxfile_free(&mtxfile);
                program_options_free(&args);
                return EXIT_FAILURE;
            }

            /* Format the output path. */
            char * output_path;
            err = format_path(
                args.mtx_output_path, &output_path, p, args.num_row_parts);
            if (err) {
                if (args.verbose > 0)
                    fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, strerror(err),
                        output_path);
                mtxfile_free(&mtxfile_p);
                free(part_per_data_line);
                free(data_lines_per_part_ptr);
                mtx_partition_free(&row_partition);
                mtxfile_free(&mtxfile);
                program_options_free(&args);
                return EXIT_FAILURE;
            }

            if (args.verbose > 0) {
                fprintf(diagf, "mtxfile_write: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }

            /* Write the part to a file. */
            setlocale(LC_ALL, "C");
            int64_t bytes_written;
            err = mtxfile_write(
                &mtxfile_p, output_path, args.gzip, args.format, &bytes_written);
            setlocale(LC_ALL, "");
            if (err) {
                if (args.verbose > 0)
                    fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtx_strerror(err));
                free(output_path);
                mtxfile_free(&mtxfile_p);
                free(part_per_data_line);
                free(data_lines_per_part_ptr);
                mtx_partition_free(&row_partition);
                mtxfile_free(&mtxfile);
                program_options_free(&args);
                return EXIT_FAILURE;
            }

            if (args.verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-6 * bytes_written / timespec_duration(t0, t1));
                fflush(diagf);
            }

            free(output_path);
            mtxfile_free(&mtxfile_p);
        }
    }

    /* 5. Write Matrix Market file for parts assigned to each matrix
     * or vector entry. */
    if (args.partition_output_path) {
        struct mtxfile mtxfile_parts;
        err = mtxfile_init_vector_array_integer_single(
            &mtxfile_parts, num_data_lines, part_per_data_line);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            free(part_per_data_line);
            free(data_lines_per_part_ptr);
            mtx_partition_free(&row_partition);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfile_comments_printf(
            &mtxfile_parts.comments, "%% This file was generated by %s %s\n",
            program_name, program_version);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtxfile_free(&mtxfile_parts);
            free(part_per_data_line);
            free(data_lines_per_part_ptr);
            mtx_partition_free(&row_partition);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        setlocale(LC_ALL, "C");
        int64_t bytes_written;
        err = mtxfile_write(
            &mtxfile_parts, args.partition_output_path,
            args.gzip, args.format, &bytes_written);
        setlocale(LC_ALL, "");
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtxfile_free(&mtxfile_parts);
            free(part_per_data_line);
            free(data_lines_per_part_ptr);
            mtx_partition_free(&row_partition);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
            fflush(diagf);
        }

        mtxfile_free(&mtxfile_parts);
    }

    /* 6. Clean up. */
    free(part_per_data_line);
    free(data_lines_per_part_ptr);
    mtx_partition_free(&row_partition);
    mtxfile_free(&mtxfile);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
