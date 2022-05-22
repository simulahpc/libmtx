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
 * Last modified: 2022-02-24
 *
 * Sort a Matrix Market file, for example, in row- or column-major
 * order.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#include <errno.h>

#include <inttypes.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxsort";
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
 * `program_options` contains data to related program options.
 */
struct program_options
{
    char * mtx_path;
    enum mtxprecision precision;
    bool gzip;
    char * format;
    char * perm_path;
    enum mtxfilesorting sorting;
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
    args->perm_path = NULL;
    args->sorting = mtxfile_row_major;
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
    if (args->perm_path)
        free(args->perm_path);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Sort a Matrix Market file.\n");
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
    fprintf(f, "  --sorting=SORTING\tsorting order: unsorted, permute, row-major,\n");
    fprintf(f, "\t\t\tcolumn-major or morton (default: row-major)\n");
    fprintf(f, "  --perm-path=FILE\tPath to output the sorting permutation,\n");
    fprintf(f, "\t\t\tunless the sorting order is ‘permute’, in which case\n");
    fprintf(f, "\t\t\tthe given path is used to read the sorting permutation.\n");
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
    fprintf(f, "%s %s (Libmtx %s)\n", program_name, program_version, libmtx_version);
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

        /* Parse output path for permutation. */
        if (strcmp((*argv)[0], "--perm-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->perm_path)
                free(args->perm_path);
            args->perm_path = strdup((*argv)[1]);
            if (!args->perm_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--perm-path=") == (*argv)[0]) {
            if (args->perm_path)
                free(args->perm_path);
            args->perm_path =
                strdup((*argv)[0] + strlen("--perm-path="));
            if (!args->perm_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Parse sorting order. */
        if (strcmp((*argv)[0], "--sorting") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = mtxfilesorting_parse(&args->sorting, NULL, NULL, (*argv)[1], NULL);
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--sorting=") == (*argv)[0]) {
            err = mtxfilesorting_parse(
                &args->sorting, NULL, NULL, (*argv)[0] + strlen("--sorting="), NULL);
            if (err) {
                program_options_free(args);
                return EINVAL;
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
    if (args.sorting == mtxfile_permutation && !args.perm_path) {
        fprintf(stderr, "%s: Please specify a sorting permutation with "
                "‘--perm-path=FILE’\n", program_invocation_short_name);
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
    int64_t lines_read = 0;
    int64_t bytes_read;
    err = mtxfile_read(
        &mtxfile, args.precision,
        args.mtx_path, args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
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

    int64_t size;
    err = mtxfilesize_num_data_lines(
        &mtxfile.size, mtxfile.header.symmetry, &size);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 2. Read the sorting permutation from a Matrix Market file, if
     * needed. Otherwise, allocate storage for outputting the sorting
     * permutation. */
    int64_t * perm = NULL;
    if (args.sorting == mtxfile_permutation) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile perm_mtxfile;
        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxfile_read(
            &perm_mtxfile, mtx_double,
            args.perm_path, false,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.perm_path, lines_read+1,
                    mtxstrerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.perm_path, mtxstrerror(err));
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
        if (perm_mtxfile.header.object != mtxfile_vector)
            err = MTX_ERR_INCOMPATIBLE_MTX_OBJECT;
        else if (perm_mtxfile.header.format != mtxfile_array)
            err = MTX_ERR_INCOMPATIBLE_MTX_FORMAT;
        else if (perm_mtxfile.header.field != mtxfile_integer)
            err = MTX_ERR_INCOMPATIBLE_MTX_FIELD;
        else if (perm_mtxfile.size.num_rows != size)
            err = MTX_ERR_INCOMPATIBLE_MTX_SIZE;
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&perm_mtxfile);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        perm = perm_mtxfile.data.array_integer_double;
        perm_mtxfile.data.array_integer_double = NULL;
        mtxfile_free(&perm_mtxfile);
    } else if (args.perm_path) {
        perm = malloc(size * sizeof(int64_t));
        if (!perm) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, strerror(errno));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_sort: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* 3. Sort the Matrix Market file. */
    err = mtxfile_sort(&mtxfile, args.sorting, size, perm);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        free(perm);
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f Mlines/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * size / timespec_duration(t0, t1));
    }

    /* 4. Write the sorted Matrix Market object to standard output. */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxfilecomments_printf(
            &mtxfile.comments,
            "%% This file was generated by %s %s (%s)\n",
            program_name, program_version, mtxfilesorting_str(args.sorting));
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(perm);
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
                    mtxstrerror(err));
            free(perm);
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

    /* Write the permutation to a Matrix Market file. */
    if (args.sorting != mtxfile_permutation && args.perm_path) {
        struct mtxfile perm_mtxfile;
        err = mtxfile_init_vector_array_integer_double(
            &perm_mtxfile, size, perm);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            free(perm);
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
            &perm_mtxfile.comments,
            "%% This file was generated by %s %s (%s)\n",
            program_name, program_version, mtxfilesorting_str(args.sorting));
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&perm_mtxfile);
            free(perm);
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        int64_t bytes_written = 0;
        err = mtxfile_write(
            &perm_mtxfile, args.perm_path, false, NULL, &bytes_written);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&perm_mtxfile);
            free(perm);
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

        mtxfile_free(&perm_mtxfile);
    }

    /* 5. Clean up. */
    free(perm);
    mtxfile_free(&mtxfile);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
