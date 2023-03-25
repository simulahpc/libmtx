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
 * Last modified: 2023-03-25
 *
 * Partition a Matrix Market file.
 */

#include <libmtx/libmtx.h>

#include "parse.h"

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
const char * program_version = LIBMTX_VERSION;
const char * program_copyright =
    "Copyright (C) 2022 James D. Trotter";
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
    char * rowpartpath;
    char * colpartpath;
    enum mtxprecision precision;
    enum mtxmatrixparttype matrixparttype;
    enum mtxpartitioning nzparttype;
    int num_nz_parts;
    int64_t nzblksize;
    enum mtxpartitioning rowparttype;
    int num_row_parts;
    int64_t rowblksize;
    enum mtxpartitioning colparttype;
    int num_column_parts;
    int64_t colblksize;
    bool gzip;
    char * format;
    bool quiet;
    int verbose;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->mtxpath = NULL;
    args->rowpartpath = NULL;
    args->colpartpath = NULL;
    args->precision = mtx_double;
    args->matrixparttype = mtx_matrixparttype_nonzeros;
    args->nzparttype = mtx_block;
    args->num_nz_parts = 1;
    args->nzblksize = 1;
    args->rowparttype = mtx_block;
    args->num_row_parts = 1;
    args->rowblksize = 1;
    args->colparttype = mtx_block;
    args->num_column_parts = 1;
    args->colblksize = 1;
    args->gzip = false;
    args->format = NULL;
    args->quiet = false;
    args->verbose = 0;
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
    if (args->rowpartpath) free(args->rowpartpath);
    if (args->colpartpath) free(args->colpartpath);
    if (args->format) free(args->format);
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
    fprintf(f, " Partition a Matrix Market file.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’ (default).\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --row-part-path=FILE\toutput path for row partitioning\n");
    fprintf(f, "  --column-part-path=FILE\toutput path for column partitioning\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  -q, --quiet\t\tdo not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
    fprintf(f, "\n");
    fprintf(f, " Options for partitioning are:\n");
    fprintf(f, "  --part-type=TYPE\tmethod of partitioning: ‘nonzeros’ (default),\n");
    fprintf(f, "\t\t\t‘rows’, ‘columns’, ‘2d’ or ‘metis’.\n");
    fprintf(f, "  --nz-parts=N\t\tnumber of parts to use when partitioning nonzeros.\n");
    fprintf(f, "  --nz-part-type=TYPE\tmethod of partitioning nonzeros if --part-type=nonzeros:\n");
    fprintf(f, "\t\t\t‘block’ (default), ‘cyclic’ or ‘block-cyclic’.\n");
    fprintf(f, "  --nz-blksize=N\tblock size to use if --nz-part-type is ‘block-cyclic’.\n");
    fprintf(f, "  --row-parts=N\t\tnumber of parts to use when partitioning rows.\n");
    fprintf(f, "  --row-part-type=TYPE\tmethod of partitioning rows if --part-type is ‘rows’ or ‘2d’:\n");
    fprintf(f, "\t\t\t‘block’ (default), ‘cyclic’ or ‘block-cyclic’.\n");
    fprintf(f, "  --row-blksize=N\tblock size to use if --row-part-type is ‘block-cyclic’.\n");
    fprintf(f, "  --column-parts=N\tnumber of parts to use when partitioning columns.\n");
    fprintf(f, "  --column-part-type=TYPE\tmethod of partitioning columns if --part-type is ‘columns’ or ‘2d’:\n");
    fprintf(f, "\t\t\t\t‘block’ (default), ‘cyclic’ or ‘block-cyclic’.\n");
    fprintf(f, "  --column-blksize=N\tblock size to use if --column-part-type is ‘block-cyclic’.\n");
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

        if (strstr(argv[0], "--row-part-path") == argv[0]) {
            int n = strlen("--row-part-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->rowpartpath) free(args->rowpartpath);
            args->rowpartpath = strdup(s);
            if (!args->rowpartpath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--column-part-path") == argv[0]) {
            int n = strlen("--column-part-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->colpartpath) free(args->colpartpath);
            args->colpartpath = strdup(s);
            if (!args->colpartpath) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--part-type") == argv[0]) {
            int n = strlen("--part-type");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxmatrixparttype(&args->matrixparttype, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--nz-parts") == argv[0]) {
            int n = strlen("--nz-parts");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->num_nz_parts, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--nz-part-type") == argv[0]) {
            int n = strlen("--nz-part-type");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxpartitioning(&args->nzparttype, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--nz-blksize") == argv[0]) {
            int n = strlen("--nz-blksize");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->nzblksize, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--row-parts") == argv[0]) {
            int n = strlen("--row-parts");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->num_row_parts, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--row-part-type") == argv[0]) {
            int n = strlen("--row-part-type");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxpartitioning(&args->rowparttype, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--row-blksize") == argv[0]) {
            int n = strlen("--row-blksize");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->rowblksize, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--column-parts") == argv[0]) {
            int n = strlen("--column-parts");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->num_column_parts, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--column-part-type") == argv[0]) {
            int n = strlen("--column-part-type");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxpartitioning(&args->colparttype, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--column-blksize") == argv[0]) {
            int n = strlen("--column-blksize");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->colblksize, s, (char **) &s, NULL);
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

    /* 2. Read a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_read(
        &mtxfile, args.precision,
        args.mtxpath, args.gzip,
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

    /* 4. partition the Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_partition: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int num_parts = 0;
    if (args.matrixparttype == mtx_matrixparttype_nonzeros) {
        num_parts = args.num_nz_parts;
    } else if (args.matrixparttype == mtx_matrixparttype_rows) {
        num_parts = args.num_row_parts;
    } else if (args.matrixparttype == mtx_matrixparttype_columns) {
        num_parts = args.num_column_parts;
    } else if (args.matrixparttype == mtx_matrixparttype_2d) {
        num_parts = args.num_row_parts*args.num_column_parts;
    } else if (args.matrixparttype == mtx_matrixparttype_metis) {
        num_parts = args.num_nz_parts;
    } else {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(MTX_ERR_INVALID_MATRIXPARTTYPE));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    int64_t num_nonzeros = mtxfile.datasize;
    int * dstnzpart = malloc(num_nonzeros * sizeof(int));
    if (!dstnzpart) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * dstnzpartsizes = malloc(num_parts * sizeof(int64_t));
    if (!dstnzpartsizes) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(dstnzpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    int64_t num_rows = mtxfile.size.num_rows;
    int * dstrowpart = malloc(num_rows * sizeof(int));
    if (!dstrowpart) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(dstnzpartsizes); free(dstnzpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * dstrowpartsizes = malloc(num_parts * sizeof(int64_t));
    if (!dstrowpartsizes) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(dstrowpart);
        free(dstnzpartsizes); free(dstnzpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    int64_t num_columns = mtxfile.size.num_columns > 0 ? mtxfile.size.num_columns : 0;
    int * dstcolpart = malloc(num_columns * sizeof(int));
    if (!dstcolpart) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(dstrowpartsizes); free(dstrowpart);
        free(dstnzpartsizes); free(dstnzpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * dstcolpartsizes = malloc(num_parts * sizeof(int64_t));
    if (!dstcolpartsizes) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(dstcolpart);
        free(dstrowpartsizes); free(dstrowpart);
        free(dstnzpartsizes); free(dstnzpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    bool rowpart, colpart;
    int64_t objval;
    err = mtxfile_partition(
        &mtxfile, args.matrixparttype,
        args.nzparttype, args.num_nz_parts, NULL, args.nzblksize, NULL,
        args.rowparttype, args.num_row_parts, NULL, args.rowblksize, NULL,
        args.colparttype, args.num_column_parts, NULL, args.colblksize, NULL,
        dstnzpart, dstnzpartsizes, &rowpart, dstrowpart, dstrowpartsizes,
        &colpart, dstcolpart, dstcolpartsizes, &objval,
        args.verbose > 2 ? args.verbose-2 : 0);
    if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, mtxstrerror(err));
        free(dstcolpartsizes); free(dstcolpart);
        free(dstrowpartsizes); free(dstrowpart);
        free(dstnzpartsizes); free(dstnzpart);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds", timespec_duration(t0, t1));
        if (args.matrixparttype == mtx_matrixparttype_metis)
            fprintf(diagf, ", edge-cut: %'"PRId64, objval);
        fprintf(diagf, "\n");
    }

    mtxfile_free(&mtxfile);

    if (args.verbose > 1) {
        fprintf(diagf, "partitioned into %'d parts:\n", num_parts);
        for (int p = 0; p < num_parts; p++) {
            fprintf(diagf, "part %'5d: %'13"PRId64" nonzeros\n",
                    p, dstnzpartsizes[p]);
        }
    }

    /* 6. write a file containing part numbers of each column. */
    if (colpart && !args.quiet && args.colpartpath) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_init_vector_array_integer_single: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile colpartsmtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &colpartsmtxfile, num_columns, dstcolpart);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            free(dstcolpartsizes); free(dstcolpart);
            free(dstrowpartsizes); free(dstrowpart);
            free(dstnzpartsizes); free(dstnzpart);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfilecomments_printf(
            &colpartsmtxfile.comments,
            "%% This file was generated by %s %s "
            "(column parts, partitioning: %s, parts: %d)\n",
            program_name, program_version,
            mtxmatrixparttypestr(args.matrixparttype), num_parts);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&colpartsmtxfile);
            free(dstcolpartsizes); free(dstcolpart);
            free(dstrowpartsizes); free(dstrowpart);
            free(dstnzpartsizes); free(dstnzpart);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fprintf(diagf, "mtxfile_fwrite: "); fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &colpartsmtxfile, args.colpartpath, args.gzip,
            args.format, &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&colpartsmtxfile);
            free(dstcolpartsizes); free(dstcolpart);
            free(dstrowpartsizes); free(dstrowpart);
            free(dstnzpartsizes); free(dstnzpart);
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
        mtxfile_free(&colpartsmtxfile);
    }
    free(dstcolpartsizes); free(dstcolpart);

    /* 6. write a file containing part numbers of each row. */
    if (rowpart && !args.quiet && args.rowpartpath) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_init_vector_array_integer_single: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile rowpartsmtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &rowpartsmtxfile, num_rows, dstrowpart);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            free(dstrowpartsizes); free(dstrowpart);
            free(dstnzpartsizes); free(dstnzpart);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfilecomments_printf(
            &rowpartsmtxfile.comments,
            "%% This file was generated by %s %s "
            "(row parts, partitioning: %s, parts: %d)\n",
            program_name, program_version,
            mtxmatrixparttypestr(args.matrixparttype), num_parts);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&rowpartsmtxfile);
            free(dstrowpartsizes); free(dstrowpart);
            free(dstnzpartsizes); free(dstnzpart);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fprintf(diagf, "mtxfile_fwrite: "); fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &rowpartsmtxfile, args.rowpartpath,
            args.gzip, args.format, &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&rowpartsmtxfile);
            free(dstrowpartsizes); free(dstrowpart);
            free(dstnzpartsizes); free(dstnzpart);
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
        mtxfile_free(&rowpartsmtxfile);
    }
    free(dstrowpartsizes); free(dstrowpart);

    /* 5. write a file containing part numbers of each nonzero. */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_init_vector_array_integer_single: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile nzpartsmtxfile;
        err = mtxfile_init_vector_array_integer_single(
            &nzpartsmtxfile, num_nonzeros, dstnzpart);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            free(dstnzpartsizes); free(dstnzpart);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfilecomments_printf(
            &nzpartsmtxfile.comments,
            "%% This file was generated by %s %s "
            "(nonzero parts, partitioning: %s, parts: %d)\n",
            program_name, program_version,
            mtxmatrixparttypestr(args.matrixparttype), num_parts);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&nzpartsmtxfile);
            free(dstnzpartsizes); free(dstnzpart);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fprintf(diagf, "mtxfile_fwrite: "); fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t bytes_written = 0;
        err = mtxfile_fwrite(&nzpartsmtxfile, stdout, args.format, &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&nzpartsmtxfile);
            free(dstnzpartsizes); free(dstnzpart);
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
        mtxfile_free(&nzpartsmtxfile);
    }

    /* 6. clean up */
    free(dstnzpartsizes); free(dstnzpart);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
