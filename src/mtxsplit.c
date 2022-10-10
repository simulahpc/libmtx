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
 * Last modified: 2022-10-08
 *
 * Split a Matrix Market file into several files.
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

const char * program_name = "mtxsplit";
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
    char * mtx_path;
    char * parts_path;
    enum mtxprecision precision;
    bool gzip;
    char * output_path;
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
    args->mtx_path = NULL;
    args->parts_path = NULL;
    args->precision = mtx_double;
    args->gzip = false;
    args->output_path = strdup("out%p.mtx");
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
    if (args->mtx_path) free(args->mtx_path);
    if (args->parts_path) free(args->parts_path);
    if (args->output_path) free(args->output_path);
    if (args->format) free(args->format);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE PARTFILE\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Split a Matrix Market file and write parts to separate files.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  FILE\t\tpath to the Matrix Market file to split\n");
    fprintf(f, "  PARTFILE\tpath to a Matrix Market file (in array format) containing\n");
    fprintf(f, "   \t\tinteger part numbers for each (nonzero) matrix or vector element\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’ (default).\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --output-path=FILE\toutput path, where occurrences of '%%p' are replaced\n");
    fprintf(f, "\t\t\twith the number of each part (default: ‘out%%p.mtx’).\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  -q, --quiet\t\tdo not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
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

        if (strstr(argv[0], "--output-path") == argv[0]) {
            int n = strlen("--output-path");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (args->output_path) free(args->output_path);
            args->output_path = strdup(s);
            if (!args->output_path) { program_options_free(args); return errno; }
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
            args->mtx_path = strdup(argv[0]);
            if (!args->mtx_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 1) {
            args->parts_path = strdup(argv[0]);
            if (!args->parts_path) { program_options_free(args); return errno; }
        } else { program_options_free(args); return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
    }

    if (num_positional_arguments_consumed < 2) {
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
 * ‘num_places()’ is the number of digits or places in a given
 * non-negative integer.
 */
static int num_places(int n, int * places)
{
    int r = 1;
    if (n < 0) return EINVAL;
    while (n > 9) { n /= 10; r++; }
    *places = r;
    return 0;
}

/**
 * ‘format_path()’ formats a path by replacing any occurence of '%p'
 * in ‘path_fmt’ with the number ‘part’.
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
        args.mtx_path, args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.mtx_path, lines_read+1,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
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

    /* 3. Read a vector of part numbers from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxparts;
    lines_read = 0;
    bytes_read = 0;
    err = mtxfile_read(
        &mtxparts, mtx_single,
        args.parts_path, args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.parts_path, lines_read+1,
                mtxstrerror(err));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.parts_path, mtxstrerror(err));
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

    /* ensure that we have an integer vector in array format */
    if (mtxparts.header.object != mtxfile_vector)
        err = MTX_ERR_INVALID_MTX_OBJECT;
    if (mtxparts.header.format != mtxfile_array)
        err = MTX_ERR_INVALID_MTX_FORMAT;
    if (mtxparts.header.field != mtxfile_integer)
        err = MTX_ERR_INVALID_MTX_FIELD;
    if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.parts_path, mtxstrerror(err));
        mtxfile_free(&mtxparts);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_split: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t size = mtxparts.size.num_rows;
    int32_t * parts = mtxparts.data.array_integer_single;
    int num_parts = 0;
    for (int64_t k = 0; k < size; k++)
        num_parts = num_parts >= parts[k]+1 ? num_parts : parts[k]+1;

    /* 4. split the Matrix Market file */
    struct mtxfile * dsts = malloc(num_parts * sizeof(struct mtxfile));
    if (!dsts) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name, strerror(errno));
        mtxfile_free(&mtxparts);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    struct mtxfile ** dstptrs = malloc(num_parts * sizeof(struct mtxfile *));
    if (!dsts) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name, strerror(errno));
        free(dsts);
        mtxfile_free(&mtxparts);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    for (int p = 0; p < num_parts; p++) dstptrs[p] = &dsts[p];

    err = mtxfile_split(num_parts, dstptrs, &mtxfile, size, parts, NULL, NULL);
    if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name, mtxstrerror(err));
        free(dstptrs);
        free(dsts);
        mtxfile_free(&mtxparts);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    free(dstptrs);
    mtxfile_free(&mtxparts);
    mtxfile_free(&mtxfile);

    /* 5. Write a Matrix Market file for each part. */
    if (args.output_path && !args.quiet) {
        for (int p = 0; p < num_parts; p++) {
            err = mtxfilecomments_printf(
                &dsts[p].comments, "%% This file was generated by %s %s (part %d of %d)\n",
                program_name, program_version, p+1, num_parts);
            if (err) {
                if (args.verbose > 0) fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                for (int q = 0; q < num_parts; q++) mtxfile_free(&dsts[q]);
                free(dsts);
                program_options_free(&args);
                return EXIT_FAILURE;
            }

            /* Format the output path. */
            char * output_path;
            err = format_path(args.output_path, &output_path, p, num_parts);
            if (err) {
                if (args.verbose > 0) fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, strerror(err),
                        output_path);
                for (int q = 0; q < num_parts; q++) mtxfile_free(&dsts[q]);
                free(dsts);
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
                &dsts[p], output_path, args.gzip,
                args.format, &bytes_written);
            if (err) {
                if (args.verbose > 0) fprintf(diagf, "\n");
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                free(output_path);
                for (int q = 0; q < num_parts; q++) mtxfile_free(&dsts[q]);
                free(dsts);
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

    /* 6. clean up */
    for (int p = 0; p < num_parts; p++) mtxfile_free(&dsts[p]);
    free(dsts);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
