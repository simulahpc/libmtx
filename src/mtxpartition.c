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
 * Last modified: 2022-01-19
 *
 * Partition a Matrix Market file and write parts to separate files.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

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
    "Copyright (C) 2022 James D. Trotter";
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
    enum mtxprecision precision;
    bool gzip;
    char * mtx_output_path;
    char * format;
    int num_row_parts;
    int num_col_parts;
    enum mtxpartitioning rowparttype;
    enum mtxpartitioning colparttype;
    char * rowpart_path;
    char * colpart_path;
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
    args->format = NULL;
    args->num_row_parts = 1;
    args->num_col_parts = 1;
    args->rowparttype = mtx_block;
    args->colparttype = mtx_block;
    args->rowpart_path = NULL;
    args->colpart_path = NULL;
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
    if (args->rowpart_path)
        free(args->rowpart_path);
    if (args->colpart_path)
        free(args->colpart_path);
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
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’. (default: ‘double’)\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --output-path=FILE\tpath for partitioned Matrix Market files, where\n");
    fprintf(f, "\t\t\t'%%p' in the string is replaced with the number\n");
    fprintf(f, "\t\t\tof each part (default: ‘out%%p.mtx’).\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  --row-parts=N\t\tnumber of parts to use when partitioning rows.\n");
    fprintf(f, "  --row-partition=TYPE\tmethod of partitioning matrix or vector rows:\n");
    fprintf(f, "\t\t\t‘block’, ‘cyclic’, ‘block-cyclic’, ‘singleton’ or ‘custom’.\n");
    fprintf(f, "\t\t\t(default: ‘block’)\n");
    fprintf(f, "  --row-partition-path=FILE\tpath to Matrix Market file for\n");
    fprintf(f, "\t\t\t\treading or writing row partition.\n");
    fprintf(f, "  --column-parts=N\t\tnumber of parts to use when partitioning columns.\n");
    fprintf(f, "  --column-partition=TYPE\tmethod of partitioning matrix or vector columns:\n");
    fprintf(f, "\t\t\t‘block’, ‘cyclic’, ‘block-cyclic’, ‘singleton’ or ‘custom’.\n");
    fprintf(f, "\t\t\t(default: ‘block’)\n");
    fprintf(f, "  --column-partition-path=FILE\tpath to Matrix Market file for\n");
    fprintf(f, "\t\t\t\treading or writing column partition.\n");
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
    fprintf(f, "%s %s (Libmtx %s)\n", program_name, program_version,
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
            err = mtxprecision_parse(&args->precision, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--precision=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--precision=");
            err = mtxprecision_parse(&args->precision, NULL, NULL, s, "");
            if (err) {
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
            err = mtxpartitioning_parse(
                &args->rowparttype, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--row-partition=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--row-partition=");
            err = mtxpartitioning_parse(
                &args->rowparttype, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }
        if (strcmp((*argv)[0], "--column-partition") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtxpartitioning_parse(
                &args->colparttype, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--column-partition=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--column-partition=");
            err = mtxpartitioning_parse(
                &args->colparttype, NULL, NULL, s, "");
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

        if (strcmp((*argv)[0], "--column-parts") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->num_col_parts, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--column-parts=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--column-parts="), NULL,
                &args->num_col_parts, NULL);
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
            if (args->rowpart_path)
                free(args->rowpart_path);
            args->rowpart_path = strdup((*argv)[1]);
            if (!args->rowpart_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--row-partition-path=") == (*argv)[0]) {
            if (args->rowpart_path)
                free(args->rowpart_path);
            args->rowpart_path =
                strdup((*argv)[0] + strlen("--row-partition-path="));
            if (!args->rowpart_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--column-partition-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->colpart_path)
                free(args->colpart_path);
            args->colpart_path = strdup((*argv)[1]);
            if (!args->colpart_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--column-partition-path=") == (*argv)[0]) {
            if (args->colpart_path)
                free(args->colpart_path);
            args->colpart_path =
                strdup((*argv)[0] + strlen("--column-partition-path="));
            if (!args->colpart_path) {
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
#ifdef LIBMTX_HAVE_MPI
            int rank;
            err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (err) {
                char mpierrstr[MPI_MAX_ERROR_STRING];
                int mpierrstrlen;
                MPI_Error_string(err, mpierrstr, &mpierrstrlen);
                fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                        program_invocation_short_name, mpierrstr);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (rank == 0)
                program_options_print_help(stdout);
            MPI_Finalize();
#else
            program_options_print_help(stdout);
#endif
            exit(EXIT_SUCCESS);
        }

        /* If requested, print program version information. */
        if (strcmp((*argv)[0], "--version") == 0) {
            program_options_free(args);
#ifdef LIBMTX_HAVE_MPI
            int rank;
            err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (err) {
                char mpierrstr[MPI_MAX_ERROR_STRING];
                int mpierrstrlen;
                MPI_Error_string(err, mpierrstr, &mpierrstrlen);
                fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                        program_invocation_short_name, mpierrstr);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (rank == 0)
                program_options_print_version(stdout);
            MPI_Finalize();
#else
            program_options_print_version(stdout);
#endif
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

#ifdef LIBMTX_HAVE_MPI
/**
 * `main()`.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    FILE * diagf = stderr;
    setlocale(LC_ALL, "");

    /* Set program invocation name. */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);

    /* Initialise MPI. */
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int root = 0;
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    struct mtxdisterror disterr;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int comm_size;
    mpierr = MPI_Comm_size(comm, &comm_size);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    err = mtxdisterror_alloc(&disterr, comm, &mpierr);
    if (err) {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxdiststrerror(err, mpierr, mpierrstr));
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 1. Parse program options. */
    struct program_options args;
    int argc_copy = argc;
    char ** argv_copy = argv;
    err = parse_program_options(&argc_copy, &argv_copy, &args);
    if (err) {
        if (rank == root) {
            fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                    strerror(err), argv_copy[0]);
        }
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank != root)
        args.verbose = 0;
    if (!args.mtx_path) {
        if (rank == root) {
            fprintf(stderr, "%s: Please specify a Matrix Market file\n",
                    program_invocation_short_name);
        }
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (args.rowparttype == mtx_custom_partition && !args.rowpart_path) {
        if (rank == root) {
            fprintf(stderr, "%s: Please specify a Matrix Market file "
                    "with --row-partition-path\n",
                    program_invocation_short_name);
        }
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (args.colparttype == mtx_custom_partition && !args.colpart_path) {
        if (rank == root) {
            fprintf(stderr, "%s: Please specify a Matrix Market file "
                    "with --column-partition-path\n",
                    program_invocation_short_name);
        }
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 2. Read a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxdistfile_read_shared: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxdistfile src;
    int lines_read;
    int64_t bytes_read;
    err = mtxdistfile_read_shared(
        &src, args.precision,
        args.mtx_path ? args.mtx_path : "", args.gzip,
        &lines_read, &bytes_read,
        comm, root, &disterr);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        if (rank == root && lines_read >= 0) {
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.mtx_path, lines_read+1,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        } else if (rank == root) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.mtx_path,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        }
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    /* 3. Create row and column partitions. */
    struct mtxpartition rowpart;
    if (args.rowparttype == mtx_custom_partition) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_read_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxpartition_read_parts(
            &rowpart, args.num_row_parts, args.rowpart_path,
            &lines_read, &bytes_read);
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root && lines_read >= 0) {
                fprintf(stderr, "%s: %s:%d: %s\n",
                        program_invocation_short_name,
                        args.rowpart_path, lines_read+1,
                        mtxdisterror_description(&disterr));
            } else if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.rowpart_path, mtxdisterror_description(&disterr));
            }
            mtxdistfile_free(&src);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
            fflush(diagf);
        }

    } else {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_init: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxpartition_init(
            &rowpart, args.rowparttype,
            src.size.num_rows, args.num_row_parts, NULL, 0, NULL, NULL);
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxdisterror_description(&disterr));
            }
            mtxdistfile_free(&src);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    struct mtxpartition colpart;
    if (args.colparttype == mtx_custom_partition) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_read_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxpartition_read_parts(
            &colpart, args.num_col_parts, args.colpart_path,
            &lines_read, &bytes_read);
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root && lines_read >= 0) {
                fprintf(stderr, "%s: %s:%d: %s\n",
                        program_invocation_short_name,
                        args.colpart_path, lines_read+1,
                        mtxdisterror_description(&disterr));
            } else if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.colpart_path, mtxdisterror_description(&disterr));
            }
            mtxpartition_free(&rowpart);
            mtxdistfile_free(&src);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
            fflush(diagf);
        }

    } else {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_init: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxpartition_init(
            &colpart, args.colparttype,
            src.size.num_columns < 0 ? 1 : src.size.num_columns,
            args.num_col_parts, NULL, 0, NULL, NULL);
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxdisterror_description(&disterr));
            }
            mtxpartition_free(&rowpart);
            mtxdistfile_free(&src);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    if (args.verbose > 0) {
        fprintf(diagf, "mtxdistfile_partition: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* 4. Partition the rows and columns of the matrix or vector. */
    int num_parts = rowpart.num_parts * colpart.num_parts;
    struct mtxdistfile * dsts = malloc(num_parts * sizeof(struct mtxdistfile));
    err = !dsts ? MTX_ERR_ERRNO : MTX_SUCCESS;
    if (mtxdisterror_allreduce(&disterr, err)) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        }
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    err = mtxdistfile_partition(
        dsts, &src, &rowpart, &colpart, &disterr);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        }
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxdistfile_free(&src);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    mtxdistfile_free(&src);

    /* 5. Write a Matrix Market file for each part. */
    if (args.mtx_output_path) {
        for (int p = 0; p < args.num_row_parts; p++) {
            for (int q = 0; q < args.num_col_parts; q++) {
                int r = p * args.num_col_parts + q;
                err = mtxfilecomments_printf(
                    &dsts[r].comments, "%% This file was generated by %s %s ("
                    "%s row partition part %d of %d, "
                    "%s column partition part %d of %d)\n",
                    program_name, program_version,
                    mtxpartitioning_str(args.rowparttype), p+1, args.num_row_parts,
                    mtxpartitioning_str(args.colparttype), q+1, args.num_col_parts);
                if (mtxdisterror_allreduce(&disterr, err)) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    if (rank == root) {
                        fprintf(stderr, "%s: %s\n",
                                program_invocation_short_name,
                                mtxdisterror_description(&disterr));
                    }
                    for (int s = 0; s < num_parts; s++)
                        mtxdistfile_free(&dsts[s]);
                    free(dsts);
                    mtxpartition_free(&colpart);
                    mtxpartition_free(&rowpart);
                    program_options_free(&args);
                    mtxdisterror_free(&disterr);
                    MPI_Finalize();
                    return EXIT_FAILURE;
                }

                /* Format the output path. */
                char * output_path;
                err = format_path(
                    args.mtx_output_path, &output_path, r, num_parts);
                if (mtxdisterror_allreduce(&disterr, err)) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    if (rank == root) {
                        fprintf(stderr, "%s: %s: %s\n",
                                program_invocation_short_name,
                                args.mtx_output_path,
                                mtxdisterror_description(&disterr));
                    }
                    for (int s = 0; s < num_parts; s++)
                        mtxdistfile_free(&dsts[s]);
                    free(dsts);
                    mtxpartition_free(&colpart);
                    mtxpartition_free(&rowpart);
                    program_options_free(&args);
                    mtxdisterror_free(&disterr);
                    MPI_Finalize();
                    return EXIT_FAILURE;
                }

                if (args.verbose > 0) {
                    fprintf(diagf, "mtxdistfile_write_shared: ");
                    fflush(diagf);
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                }

                /* Write the part to a file. */
                int64_t bytes_written = 0;
                err = mtxdistfile_write_shared(
                    &dsts[r], output_path, args.gzip,
                    args.format, &bytes_written,
                    root, &disterr);
                if (err) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    if (rank == root) {
                        fprintf(stderr, "%s: %s\n",
                                program_invocation_short_name,
                                err == MTX_ERR_MPI_COLLECTIVE
                                ? mtxdisterror_description(&disterr)
                                : mtxstrerror(err));
                    }
                    free(output_path);
                    for (int s = 0; s < num_parts; s++)
                        mtxdistfile_free(&dsts[s]);
                    free(dsts);
                    mtxpartition_free(&colpart);
                    mtxpartition_free(&rowpart);
                    program_options_free(&args);
                    mtxdisterror_free(&disterr);
                    MPI_Finalize();
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
            }
        }
    }

    for (int r = 0; r < num_parts; r++)
        mtxdistfile_free(&dsts[r]);
    free(dsts);

    /* 6. Write a Matrix Market file containing the part numbers
     * assigned to each row of the matrix or vector. */
    if (args.rowparttype != mtx_custom_partition && args.rowpart_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_write_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = (rank == root) ? mtxpartition_write_parts(
            &rowpart, args.rowpart_path, "%d", &bytes_written)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.rowpart_path,
                        mtxdisterror_description(&disterr));
            }
            mtxpartition_free(&colpart);
            mtxpartition_free(&rowpart);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 7. Write a Matrix Market file containing the part numbers
     * assigned to each column of the matrix or vector. */
    if (args.colparttype != mtx_custom_partition && args.colpart_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_write_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = (rank == root) ? mtxpartition_write_parts(
            &colpart, args.colpart_path, "%d", &bytes_written)
            : MTX_SUCCESS;
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.colpart_path,
                        mtxdisterror_description(&disterr));
            }
            mtxpartition_free(&colpart);
            mtxpartition_free(&rowpart);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 8. Clean up. */
    mtxpartition_free(&colpart);
    mtxpartition_free(&rowpart);
    program_options_free(&args);
    mtxdisterror_free(&disterr);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
#else
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
    if (args.rowparttype == mtx_custom_partition && !args.rowpart_path) {
        fprintf(stderr, "%s: Please specify a Matrix Market file "
                "with --row-partition-path\n",
                program_invocation_short_name);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (args.colparttype == mtx_custom_partition && !args.colpart_path) {
        fprintf(stderr, "%s: Please specify a Matrix Market file "
                "with --column-partition-path\n",
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
    err = mtxfile_read(
        &mtxfile, args.precision,
        args.mtx_path, args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d: %s\n",
                program_invocation_short_name,
                args.mtx_path, lines_read+1,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.mtx_path, mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    /* 3. Create row and column partitions. */
    struct mtxpartition rowpart;
    if (args.rowparttype == mtx_custom_partition) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_read_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxpartition_read_parts(
            &rowpart, args.num_row_parts, args.rowpart_path,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.rowpart_path, lines_read+1,
                    mtxstrerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.rowpart_path, mtxstrerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
            fflush(diagf);
        }

    } else {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_init: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxpartition_init(
            &rowpart, args.rowparttype,
            mtxfile.size.num_rows, args.num_row_parts, NULL, 0, NULL, NULL);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    struct mtxpartition colpart;
    if (args.colparttype == mtx_custom_partition) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_read_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxpartition_read_parts(
            &colpart, args.num_col_parts, args.colpart_path,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.colpart_path, lines_read+1,
                    mtxstrerror(err));
            mtxpartition_free(&rowpart);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.colpart_path, mtxstrerror(err));
            mtxpartition_free(&rowpart);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
            fflush(diagf);
        }

    } else {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_init: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxpartition_init(
            &colpart, args.colparttype,
            mtxfile.size.num_columns, args.num_col_parts, NULL, 0, NULL, NULL);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxpartition_free(&rowpart);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 4. Partition the rows and columns of the matrix or vector. */
    int num_parts = rowpart.num_parts * colpart.num_parts;
    struct mtxfile * dsts = malloc(num_parts * sizeof(struct mtxfile));
    if (!dsts) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name, strerror(errno));
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_partition: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    err = mtxfile_partition(
        dsts, &mtxfile, &rowpart, &colpart);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name, mtxstrerror(err));
        free(dsts);
        mtxpartition_free(&colpart);
        mtxpartition_free(&rowpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    mtxfile_free(&mtxfile);

    /* 5. Write a Matrix Market file for each part. */
    if (args.mtx_output_path) {
        for (int p = 0; p < args.num_row_parts; p++) {
            for (int q = 0; q < args.num_col_parts; q++) {
                int r = p * args.num_col_parts + q;
                err = mtxfilecomments_printf(
                    &dsts[r].comments, "%% This file was generated by %s %s ("
                    "%s row partition part %d of %d, "
                    "%s column partition part %d of %d)\n",
                    program_name, program_version,
                    mtxpartitioning_str(args.rowparttype), p+1, args.num_row_parts,
                    mtxpartitioning_str(args.colparttype), q+1, args.num_col_parts);
                if (err) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name,
                            mtxstrerror(err));
                    for (int s = 0; s < num_parts; s++)
                        mtxfile_free(&dsts[s]);
                    free(dsts);
                    mtxpartition_free(&colpart);
                    mtxpartition_free(&rowpart);
                    program_options_free(&args);
                    return EXIT_FAILURE;
                }

                /* Format the output path. */
                char * output_path;
                err = format_path(
                    args.mtx_output_path, &output_path, r, num_parts);
                if (err) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    fprintf(stderr, "%s: %s: %s\n",
                            program_invocation_short_name, strerror(err),
                            output_path);
                    for (int s = 0; s < num_parts; s++)
                        mtxfile_free(&dsts[s]);
                    free(dsts);
                    mtxpartition_free(&colpart);
                    mtxpartition_free(&rowpart);
                    program_options_free(&args);
                    return EXIT_FAILURE;
                }

                if (args.verbose > 0) {
                    fprintf(diagf, "mtxfile_write: ");
                    fflush(diagf);
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                }

                /* Write the part to a file. */
                int64_t bytes_written = 0;
                err = mtxfile_write(
                    &dsts[r], output_path, args.gzip,
                    args.format, &bytes_written);
                if (err) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name,
                            mtxstrerror(err));
                    free(output_path);
                    for (int s = 0; s < num_parts; s++)
                        mtxfile_free(&dsts[s]);
                    free(dsts);
                    mtxpartition_free(&colpart);
                    mtxpartition_free(&rowpart);
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
            }
        }
    }

    for (int r = 0; r < num_parts; r++)
        mtxfile_free(&dsts[r]);
    free(dsts);

    /* 6. Write a Matrix Market file containing the part numbers
     * assigned to each row of the matrix or vector. */
    if (args.rowparttype != mtx_custom_partition && args.rowpart_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_write_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = mtxpartition_write_parts(
            &rowpart, args.rowpart_path, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.rowpart_path,
                    mtxstrerror(err));
            mtxpartition_free(&colpart);
            mtxpartition_free(&rowpart);
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
    }

    /* 7. Write a Matrix Market file containing the part numbers
     * assigned to each column of the matrix or vector. */
    if (args.colparttype != mtx_custom_partition && args.colpart_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_write_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = mtxpartition_write_parts(
            &colpart, args.colpart_path, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.colpart_path,
                    mtxstrerror(err));
            mtxpartition_free(&colpart);
            mtxpartition_free(&rowpart);
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
    }

    /* 8. Clean up. */
    mtxpartition_free(&colpart);
    mtxpartition_free(&rowpart);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
#endif
