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
 * Reorder the nonzeros of a sparse matrix and any number of vectors
 * in Matrix Market format according to a specified reordering
 * algorithm or a given permutation.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#include <errno.h>

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxreorder";
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
    char * format;
    char * rowperm_path;
    char * colperm_path;
    enum mtxfile_ordering ordering;
    int rcm_starting_vertex;
    int verbose;
    bool quiet;
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
    args->format = NULL;
    args->rowperm_path = NULL;
    args->colperm_path = NULL;
    args->ordering = mtxfile_rcm;
    args->rcm_starting_vertex = 0;
    args->verbose = 0;
    args->quiet = false;
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
    if (args->format)
        free(args->format);
    if (args->rowperm_path)
        free(args->rowperm_path);
    if (args->colperm_path)
        free(args->colperm_path);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Reorder rows and columns of a matrix in Matrix Market format.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: single or double. (default: double)\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter the file through gzip\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  --rowperm-path=FILE\tpath for outputting row permuation\n");
    fprintf(f, "  --colperm-path=FILE\tpath for outputting column permuation\n");
    fprintf(f, "  --ordering=ORDERING\tordering to use: rcm (default: rcm).\n");
    fprintf(f, "  --rcm-starting-vertex=N\tstarting vertex for the RCM algorithm.\n");
    fprintf(f, "\t\t\tThe default value is 0, which means to choose automatically.\n");
    fprintf(f, "  -q, --quiet\t\tdo not print Matrix Market output\n");
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
            err = mtx_precision_parse(&args->precision, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--precision=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--precision=");
            err = mtx_precision_parse(&args->precision, NULL, NULL, s, "");
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

        /* Parse output path for row permutation. */
        if (strcmp((*argv)[0], "--rowperm-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->rowperm_path)
                free(args->rowperm_path);
            args->rowperm_path = strdup((*argv)[1]);
            if (!args->rowperm_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--rowperm-path=") == (*argv)[0]) {
            if (args->rowperm_path)
                free(args->rowperm_path);
            args->rowperm_path =
                strdup((*argv)[0] + strlen("--rowperm-path="));
            if (!args->rowperm_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Parse output path for column permutation. */
        if (strcmp((*argv)[0], "--colperm-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->colperm_path)
                free(args->colperm_path);
            args->colperm_path = strdup((*argv)[1]);
            if (!args->colperm_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--colperm-path=") == (*argv)[0]) {
            if (args->colperm_path)
                free(args->colperm_path);
            args->colperm_path =
                strdup((*argv)[0] + strlen("--colperm-path="));
            if (!args->colperm_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Parse ordering. */
        if (strcmp((*argv)[0], "--ordering") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            if (strcmp(s, "rcm") == 0) {
                args->ordering = mtxfile_rcm;
            } else {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--ordering=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--ordering=");
            if (strcmp(s, "rcm") == 0) {
                args->ordering = mtxfile_rcm;
            } else {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--rcm-starting-vertex") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->rcm_starting_vertex, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--rcm-starting-vertex=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--rcm-starting-vertex="), NULL,
                &args->rcm_starting_vertex, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "-q") == 0 || strcmp((*argv)[0], "--quiet") == 0) {
            args->quiet = true;
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
 * `main()`.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    FILE * diagf = stderr;

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
        &mtxfile, args.precision, args.mtx_path, args.gzip,
        &lines_read, &bytes_read);
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

    /* 3. Reorder the matrix. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_reorder: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    err = MTX_SUCCESS;
    if (mtxfile.header.object != mtxfile_matrix)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    else if (mtxfile.header.format != mtxfile_coordinate)
        err = MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtx_strerror(err));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    int num_rows = mtxfile.size.num_rows;
    int num_columns = mtxfile.size.num_columns;
    int * rowperm = malloc((num_rows + num_columns) * sizeof(int));
    if (!rowperm) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtx_strerror(MTX_ERR_ERRNO));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int * colperm = &rowperm[num_rows];

    int rcm_starting_vertex = args.rcm_starting_vertex;
    err = mtxfile_reorder(
        &mtxfile, args.ordering, rowperm, colperm,
        &rcm_starting_vertex);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtx_strerror(err));
        free(rowperm);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 4. Write the reordered matrix to standard output. */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.ordering == mtxfile_rcm) {
            err = mtxfile_comments_printf(
                &mtxfile.comments, "%% This file was generated by %s %s (%s, starting vertex %d)\n",
                program_name, program_version, mtxfile_ordering_str(args.ordering), rcm_starting_vertex);
        } else {
            err = mtxfile_comments_printf(
                &mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
                program_name, program_version, mtxfile_ordering_str(args.ordering));
        }
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_fwrite(&mtxfile, stdout, args.format, &bytes_written);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            free(rowperm);
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
    }

    /* Write the row permutation to a Matrix Market file. */
    if (args.rowperm_path) {
        struct mtxfile rowperm_mtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &rowperm_mtxfile, mtxfile.size.num_rows, rowperm);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.ordering == mtxfile_rcm) {
            err = mtxfile_comments_printf(
                &rowperm_mtxfile.comments, "%% This file was generated by %s %s (%s, starting vertex %d)\n",
                program_name, program_version, mtxfile_ordering_str(args.ordering), rcm_starting_vertex);
        } else {
            err = mtxfile_comments_printf(
                &rowperm_mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
                program_name, program_version, mtxfile_ordering_str(args.ordering));
        }
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtxfile_free(&rowperm_mtxfile);
            free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &rowperm_mtxfile, args.rowperm_path, false, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtxfile_free(&rowperm_mtxfile);
            free(rowperm);
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

        mtxfile_free(&rowperm_mtxfile);
    }

    /* Write the column permutation to a Matrix Market file. */
    if (args.colperm_path) {
        struct mtxfile colperm_mtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &colperm_mtxfile, mtxfile.size.num_columns, colperm);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.ordering == mtxfile_rcm) {
            err = mtxfile_comments_printf(
                &colperm_mtxfile.comments, "%% This file was generated by %s %s (%s, starting vertex %d)\n",
                program_name, program_version, mtxfile_ordering_str(args.ordering), rcm_starting_vertex);
        } else {
            err = mtxfile_comments_printf(
                &colperm_mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
                program_name, program_version, mtxfile_ordering_str(args.ordering));
        }
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtxfile_free(&colperm_mtxfile);
            free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &colperm_mtxfile, args.colperm_path, false, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtxfile_free(&colperm_mtxfile);
            free(rowperm);
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
        mtxfile_free(&colperm_mtxfile);
    }

    /* 5. Clean up. */
    free(rowperm);
    mtxfile_free(&mtxfile);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
