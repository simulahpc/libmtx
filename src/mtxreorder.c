/* This file is part of Libmtx.
 *
 * Copyright (C) 2023 James D. Trotter
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
 * Last modified: 2023-03-24
 *
 * Reorder the nonzeros of a sparse matrix and any number of vectors
 * in Matrix Market format according to a specified reordering
 * algorithm or a given permutation.
 */

#include <libmtx/libmtx.h>

#include "parse.h"

#include <errno.h>

#include <inttypes.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxreorder";
const char * program_version = LIBMTX_VERSION;
const char * program_copyright =
    "Copyright (C) 2023 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    char * mtxpath;
    enum mtxprecision precision;
    bool gzip;
    char * format;
    char * rowpermpath;
    char * rowperminvpath;
    char * colpermpath;
    char * colperminvpath;
    enum mtxfileordering ordering;
    int64_t rcm_starting_vertex;
    int nparts;
    char * rowpartsizespath;
    char * colpartsizespath;
    int verbose;
    bool quiet;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->mtxpath = NULL;
    args->precision = mtx_double;
    args->gzip = false;
    args->format = NULL;
    args->rowpermpath = NULL;
    args->rowperminvpath = NULL;
    args->colpermpath = NULL;
    args->colperminvpath = NULL;
    args->ordering = mtxfile_rcm;
    args->rcm_starting_vertex = 0;
    args->nparts = 2;
    args->rowpartsizespath = NULL;
    args->colpartsizespath = NULL;
    args->verbose = 0;
    args->quiet = false;
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->mtxpath) free(args->mtxpath);
    if (args->format) free(args->format);
    if (args->rowpermpath) free(args->rowpermpath);
    if (args->rowperminvpath) free(args->rowperminvpath);
    if (args->colpermpath) free(args->colpermpath);
    if (args->colperminvpath) free(args->colperminvpath);
    if (args->rowpartsizespath) free(args->rowpartsizespath);
    if (args->colpartsizespath) free(args->colpartsizespath);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Reorder rows and columns of a matrix in Matrix Market format.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’ (default).\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  -q, --quiet\t\tdo not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
    fprintf(f, "\n");
    fprintf(f, " Options for reordering are:\n");
    fprintf(f, "  --ordering=ORDERING\tordering to use: default, custom, metis, nd, rcm. [rcm]\n");
    fprintf(f, "  --rowperm-path=FILE\tPath for outputting row permutation, unless the ordering\n");
    fprintf(f, "\t\t\tis ‘custom’, in which case the path is used to specify\n");
    fprintf(f, "\t\t\ta row permutation to apply.\n");
    fprintf(f, "  --colperm-path=FILE\tPath for outputting column permutation, unless the ordering\n");
    fprintf(f, "\t\t\tis ‘custom’, in which case the path is used to specify\n");
    fprintf(f, "\t\t\ta column permutation to apply.\n");
    fprintf(f, "  --rowperm-inv-path=FILE\tpath for outputting inverse row permutation\n");
    fprintf(f, "  --colperm-inv-path=FILE\tpath for outputting inverse column permutation\n");
    fprintf(f, "\n");
    fprintf(f, " Options for RCM are:\n");
    fprintf(f, "  --rcm-starting-vertex=N\tStarting vertex for the RCM algorithm.\n");
    fprintf(f, "\t\t\tThe default value is 0, which means to choose automatically.\n");
    fprintf(f, "\n");
    fprintf(f, " Options for METIS are:\n");
    fprintf(f, "  --parts=N\tnumber of parts in partitioning. [2]\n");
    fprintf(f, "  --rowpartsize-path=FILE\tpath for outputting size of row partitions\n");
    fprintf(f, "  --colpartsize-path=FILE\tpath for outputting size of column partitions\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help\t\tdisplay this help and exit\n");
    fprintf(f, "  --version\t\tdisplay version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s (Libmtx %s)\n", program_name, program_version, libmtx_version);
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * ‘parse_program_options()’ parses program options.
 */
static int parse_program_options(
    int argc,
    char ** argv,
    struct program_options * args,
    int * nargs)
{
    int err;
    *nargs = 0;

    /* Set program invocation name. */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);
    (*nargs)++; argv++;

    /* Set default program options. */
    err = program_options_init(args);
    if (err)
        return err;

    /* Parse program options. */
    int num_positional_arguments_consumed = 0;
    while (*nargs < argc) {
        if (strstr(argv[0], "--precision") == argv[0]) {
            int n = strlen("--precision");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxprecision(&args->precision, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-z") == 0 ||
            strcmp(argv[0], "--gzip") == 0 ||
            strcmp(argv[0], "--gunzip") == 0 ||
            strcmp(argv[0], "--ungzip") == 0)
        {
            args->gzip = true;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--format") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            args->format = strdup(argv[0]);
            if (!args->format) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--format=") == argv[0]) {
            args->format = strdup(argv[0] + strlen("--format="));
            if (!args->format) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = true;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--ordering") == argv[0]) {
            int n = strlen("--ordering");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxfileordering(&args->ordering, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--rowperm-path") == argv[0]) {
            int n = strlen("--rowperm-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->rowpermpath) free(args->rowpermpath);
            args->rowpermpath = strdup(s);
            if (!args->rowpermpath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--rowperm-inv-path") == argv[0]) {
            int n = strlen("--rowperm-inv-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->rowperminvpath) free(args->rowperminvpath);
            args->rowperminvpath = strdup(s);
            if (!args->rowperminvpath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--colperm-path") == argv[0]) {
            int n = strlen("--colperm-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->colpermpath) free(args->colpermpath);
            args->colpermpath = strdup(s);
            if (!args->colpermpath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--colperm-inv-path") == argv[0]) {
            int n = strlen("--colperm-inv-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->colperminvpath) free(args->colperminvpath);
            args->colperminvpath = strdup(s);
            if (!args->colperminvpath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--rcm-starting-vertex") == argv[0]) {
            int n = strlen("--rcm-starting-vertex");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->rcm_starting_vertex, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--parts") == argv[0]) {
            int n = strlen("--parts");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->nparts, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--rowpartsize-path") == argv[0]) {
            int n = strlen("--rowpartsize-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->rowpartsizespath) free(args->rowpartsizespath);
            args->rowpartsizespath = strdup(s);
            if (!args->rowpartsizespath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--colpartsize-path") == argv[0]) {
            int n = strlen("--colpartsize-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->colpartsizespath) free(args->colpartsizespath);
            args->colpartsizespath = strdup(s);
            if (!args->colpartsizespath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        /* If requested, print program help text. */
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0) {
            program_options_free(args);
            program_options_print_help(stdout);
            exit(EXIT_SUCCESS);
        }

        /* If requested, print program version information. */
        if (strcmp(argv[0], "--version") == 0) {
            program_options_free(args);
            program_options_print_version(stdout);
            exit(EXIT_SUCCESS);
        }

        /* Stop parsing options after '--'.  */
        if (strcmp(argv[0], "--") == 0) {
            (*nargs)++; argv++;
            break;
        }

        /* Unrecognised option. */
        if (strlen(argv[0]) > 1 && argv[0][0] == '-' &&
            ((argv[0][1] < '0' || argv[0][1] > '9') && argv[0][1] != '.'))
        {
            program_options_free(args);
            return EINVAL;
        }

        /*
         * Parse positional arguments.
         */
        if (num_positional_arguments_consumed == 0) {
            args->mtxpath = strdup(argv[0]);
            if (!args->mtxpath) { program_options_free(args); return errno; }
        } else { program_options_free(args); return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
    }

    if (num_positional_arguments_consumed < 1) {
        program_options_free(args);
        program_options_print_usage(stdout);
        exit(EXIT_FAILURE);
    }
    return 0;
}

/**
 * ‘timespec_duration()’ is the duration, in seconds, elapsed between
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
 * ‘main()’.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    FILE * diagf = stderr;
    setlocale(LC_ALL, "");

    /* set program invocation name */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);

    /* 1. parse program options */
    struct program_options args;
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        return EXIT_FAILURE;
    }

    if (args.ordering == mtxfile_custom_order &&
        !args.rowpermpath && !args.colpermpath)
    {
        fprintf(stderr, "%s: Please specify a row or column permutation with "
                "‘--rowperm-path=FILE’ or ‘--colperm-path=FILE’\n",
                program_invocation_short_name);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 2. read a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_read(
        &mtxfile, args.precision, args.mtxpath, args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.mtxpath, lines_read+1,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.mtxpath, mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    err = MTX_SUCCESS;
    if (mtxfile.header.object != mtxfile_matrix)
        err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
    else if (mtxfile.header.format != mtxfile_coordinate)
        err = MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
    if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 3. Read row and column permutations from Matrix Market files,
     * if needed. Otherwise, allocate storage for outputting row and
     * column permutations. */
    int num_rows = mtxfile.size.num_rows;
    int num_columns = mtxfile.size.num_columns;
    int32_t * rowperm = NULL;
    int32_t * colperm = NULL;
    if (args.ordering == mtxfile_custom_order && args.rowpermpath) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile rowperm_mtxfile;
        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxfile_read(
            &rowperm_mtxfile, mtx_single,
            args.rowpermpath, args.gzip,
            &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (lines_read >= 0) {
                fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                        program_invocation_short_name,
                        args.rowpermpath, lines_read+1,
                        mtxstrerror(err));
            } else {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.rowpermpath, mtxstrerror(err));
            }
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

        err = MTX_SUCCESS;
        if (rowperm_mtxfile.header.object != mtxfile_vector)
            err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
        else if (rowperm_mtxfile.header.format != mtxfile_array)
            err = MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
        else if (rowperm_mtxfile.header.field != mtxfile_integer)
            err = MTX_ERR_INCOMPATIBLE_MTX_FIELD;
        else if (rowperm_mtxfile.size.num_rows != mtxfile.size.num_rows)
            err = MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&rowperm_mtxfile);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        rowperm = rowperm_mtxfile.data.array_integer_single;
        rowperm_mtxfile.data.array_integer_single = NULL;
        mtxfile_free(&rowperm_mtxfile);
    } else if (args.rowpermpath) {
        rowperm = malloc(num_rows * sizeof(int32_t));
        if (!rowperm) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_ERRNO));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    if (args.ordering == mtxfile_custom_order && args.colpermpath) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile colperm_mtxfile;
        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxfile_read(
            &colperm_mtxfile, mtx_single,
            args.colpermpath, args.gzip,
            &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (lines_read >= 0) {
                fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                        program_invocation_short_name,
                        args.colpermpath, lines_read+1,
                        mtxstrerror(err));
            } else {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.colpermpath, mtxstrerror(err));
            }
            free(rowperm);
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

        err = MTX_SUCCESS;
        if (colperm_mtxfile.header.object != mtxfile_vector)
            err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
        else if (colperm_mtxfile.header.format != mtxfile_array)
            err = MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
        else if (colperm_mtxfile.header.field != mtxfile_integer)
            err = MTX_ERR_INCOMPATIBLE_MTX_FIELD;
        else if (colperm_mtxfile.size.num_rows != mtxfile.size.num_columns)
            err = MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(rowperm);
            mtxfile_free(&colperm_mtxfile);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        colperm = colperm_mtxfile.data.array_integer_single;
        colperm_mtxfile.data.array_integer_single = NULL;
        mtxfile_free(&colperm_mtxfile);
    } else if (args.colpermpath) {
        colperm = malloc(num_columns * sizeof(int32_t));
        if (!colperm) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_ERRNO));
            free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    int32_t * rowperminv = NULL;
    int32_t * colperminv = NULL;
    if (args.rowperminvpath) {
        rowperminv = malloc(num_rows * sizeof(int32_t));
        if (!rowperminv) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_ERRNO));
            free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
    if (args.colperminvpath) {
        colperminv = malloc(num_columns * sizeof(int32_t));
        if (!colperminv) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_ERRNO));
            free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    int32_t * rowpartsizes = NULL;
    int32_t * colpartsizes = NULL;
    if (args.ordering == mtxfile_metis && args.rowpartsizespath) {
        rowpartsizes = malloc(args.nparts * sizeof(int32_t));
        if (!rowpartsizes) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_ERRNO));
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
    if (args.ordering == mtxfile_metis && args.colpartsizespath) {
        colpartsizes = malloc(args.nparts * sizeof(int32_t));
        if (!colpartsizes) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_ERRNO));
            free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 3. Reorder the matrix. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_reorder: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    bool symmetric;
    int rcm_starting_vertex = args.rcm_starting_vertex;
    err = mtxfile_reorder(
        &mtxfile, args.ordering, rowperm, rowperminv, colperm, colperminv,
        !args.quiet, &symmetric, &rcm_starting_vertex,
        args.nparts, rowpartsizes, colpartsizes,
        args.verbose > 1 ? args.verbose-1 : 0);
    if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        free(colpartsizes); free(rowpartsizes);
        free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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
            err = mtxfilecomments_printf(
                &mtxfile.comments, "%% This file was generated by %s %s (%s, starting vertex %'d)\n",
                program_name, program_version, mtxfileorderingstr(args.ordering), rcm_starting_vertex);
        } else {
            err = mtxfilecomments_printf(
                &mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
                program_name, program_version, mtxfileorderingstr(args.ordering));
        }
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_fwrite(&mtxfile, stdout, args.format, &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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

    /* 5. Write the row permutation to a Matrix Market file. */
    if (args.ordering != mtxfile_custom_order && args.rowpermpath) {
        struct mtxfile rowperm_mtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &rowperm_mtxfile, mtxfile.size.num_rows, rowperm);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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
            err = mtxfilecomments_printf(
                &rowperm_mtxfile.comments, "%% This file was generated by %s %s (%s, starting vertex %'d)\n",
                program_name, program_version, mtxfileorderingstr(args.ordering), rcm_starting_vertex);
        } else {
            err = mtxfilecomments_printf(
                &rowperm_mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
                program_name, program_version, mtxfileorderingstr(args.ordering));
        }
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&rowperm_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &rowperm_mtxfile, args.rowpermpath, false, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&rowperm_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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

    /* 6. Write the column permutation to a Matrix Market file. */
    if (args.ordering != mtxfile_custom_order && args.colpermpath && !symmetric) {
        struct mtxfile colperm_mtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &colperm_mtxfile, mtxfile.size.num_columns, colperm);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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
            err = mtxfilecomments_printf(
                &colperm_mtxfile.comments, "%% This file was generated by %s %s (%s, starting vertex %'d)\n",
                program_name, program_version, mtxfileorderingstr(args.ordering), rcm_starting_vertex);
        } else {
            err = mtxfilecomments_printf(
                &colperm_mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
                program_name, program_version, mtxfileorderingstr(args.ordering));
        }
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&colperm_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &colperm_mtxfile, args.colpermpath, false, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&colperm_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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

    /* 7. write size of row partitions to a Matrix Market file */
    if (args.ordering == mtxfile_metis && args.rowpartsizespath) {
        struct mtxfile rowpartsizes_mtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &rowpartsizes_mtxfile, args.nparts, rowpartsizes);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxfilecomments_printf(
            &rowpartsizes_mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
            program_name, program_version, mtxfileorderingstr(args.ordering));
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&rowpartsizes_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &rowpartsizes_mtxfile, args.rowpartsizespath, false, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&rowpartsizes_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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
        mtxfile_free(&rowpartsizes_mtxfile);
    }

    /* 7. write size of column partitions to a Matrix Market file */
    if (args.ordering == mtxfile_metis && args.colpartsizespath && !symmetric) {
        struct mtxfile colpartsizes_mtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &colpartsizes_mtxfile, args.nparts, colpartsizes);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxfilecomments_printf(
            &colpartsizes_mtxfile.comments, "%% This file was generated by %s %s (%s)\n",
            program_name, program_version, mtxfileorderingstr(args.ordering));
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&colpartsizes_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &colpartsizes_mtxfile, args.colpartsizespath, false, "%d", &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&colpartsizes_mtxfile);
            free(colpartsizes); free(rowpartsizes);
            free(colperminv); free(rowperminv); free(colperm); free(rowperm);
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
        mtxfile_free(&colpartsizes_mtxfile);
    }

    /* 7. clean up. */
    free(colpartsizes); free(rowpartsizes);
    free(colperminv); free(rowperminv); free(colperm); free(rowperm);
    mtxfile_free(&mtxfile);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
