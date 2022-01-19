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
 * Last modified: 2022-01-19
 *
 * Multiply a general, unsymmetric matrix with a vector.
 *
 * ‘y := alpha*A*x + beta*y’.
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

const char * program_name = "mtxgemv";
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
    double alpha;
    double beta;
    char * A_path;
    char * x_path;
    char * y_path;
    char * format;
    enum mtxprecision precision;
    enum mtxmatrixtype matrix_type;
    enum mtxvectortype vector_type;
    enum mtxfield vector_field;
    bool gzip;
    int repeat;
    int verbose;
    bool quiet;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->alpha = 1.0;
    args->beta = 1.0;
    args->A_path = NULL;
    args->x_path = NULL;
    args->y_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->matrix_type = mtxmatrix_auto;
    args->vector_type = mtxvector_auto;
    args->vector_field = mtx_field_real;
    args->gzip = false;
    args->repeat = 1;
    args->quiet = false;
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
    if (args->A_path)
        free(args->A_path);
    if (args->x_path)
        free(args->x_path);
    if (args->y_path)
        free(args->y_path);
    if(args->format)
        free(args->format);
}

/**
 * `program_options_print_usage()` prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] alpha A [x] [beta] [y]\n", program_name);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Multiply a general, unsymmetric matrix with a vector.\n");
    fprintf(f, "\n");
    fprintf(f, " The operation performed is `y := alpha*A*x + beta*y',\n");
    fprintf(f, " where `A' is a matrix, `x' and `y' are vectors, and\n");
    fprintf(f, " `alpha' and `beta' are scalar, floating-point values.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  alpha\tScalar floating-point value.\n");
    fprintf(f, "  A\tPath to Matrix Market file for the matrix A.\n");
    fprintf(f, "  x\tOptional path to Matrix Market file for the vector x.\n");
    fprintf(f, "   \tIf omitted, then a vector of ones is used.\n");
    fprintf(f, "  beta\tOptional scalar floating-point value (default: 1.0).\n");
    fprintf(f, "  y\tOptional path to Matrix Market file for the vector y.\n");
    fprintf(f, "   \tIf omitted, then a vector of zeros is used.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix\n");
    fprintf(f, "\t\t\tvector values: single or double. (default: double)\n");
    fprintf(f, "  --matrix-type=TYPE\tformat for representing matrices:\n");
    fprintf(f, "\t\t\t‘auto’, ‘array’ or ‘coordinate’. (default: ‘auto’)\n");
    fprintf(f, "  --vector-type=TYPE\tformat for representing vectors:\n");
    fprintf(f, "\t\t\t‘auto’, ‘array’ or ‘coordinate’. (default: ‘auto’)\n");
    fprintf(f, "  --vector-field=TYPE\tIf the matrix is in coordinate format with ‘pattern’ field,\n");
    fprintf(f, "\t\t\tthen this is used to select the field for vector values:\n");
    fprintf(f, "\t\t\t‘real’, ‘complex’ or ‘integer’. (default: ‘real’)\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  --repeat=N\t\trepeat matrix-vector multiplication N times\n");
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
    int num_positional_arguments_consumed = 0;
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

        if (strcmp((*argv)[0], "--matrix-type") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtxmatrixtype_parse(
                &args->matrix_type, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--matrix-type=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--matrix-type=");
            err = mtxmatrixtype_parse(
                &args->matrix_type, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--vector-type") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtxvectortype_parse(
                &args->vector_type, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--vector-type=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--vector-type=");
            err = mtxvectortype_parse(
                &args->vector_type, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--vector-field") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtxfield_parse(&args->vector_field, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--vector-field=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--vector-field=");
            err = mtxfield_parse(&args->vector_field, NULL, NULL, s, "");
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

        /* Parse number of repeats. */
        if (strcmp((*argv)[0], "--repeat") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->repeat, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--repeat=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--repeat="), NULL,
                &args->repeat, NULL);
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

        /*
         * Parse positional arguments.
         */
        if (num_positional_arguments_consumed == 0) {
            err = parse_double((*argv)[0], NULL, &args->alpha, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
        } else if (num_positional_arguments_consumed == 1) {
            args->A_path = strdup((*argv)[0]);
            if (!args->A_path) {
                program_options_free(args);
                return errno;
            }
        } else if (num_positional_arguments_consumed == 2) {
            args->x_path = strdup((*argv)[0]);
            if (!args->x_path) {
                program_options_free(args);
                return errno;
            }
        } else if (num_positional_arguments_consumed == 3) {
            err = parse_double((*argv)[0], NULL, &args->beta, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
        } else if (num_positional_arguments_consumed == 4) {
            args->y_path = strdup((*argv)[0]);
            if (!args->y_path) {
                program_options_free(args);
                return errno;
            }
        } else {
            program_options_free(args);
            return EINVAL;
        }

        num_positional_arguments_consumed++;
        num_arguments_consumed++;
    }

    if (num_positional_arguments_consumed < 2) {
        program_options_free(args);
        program_options_print_usage(stdout);
        exit(EXIT_FAILURE);
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
 * `gemv()' computes matrix-vector products and prints the result to
 * standard output.
 */
static int gemv(
    double alpha,
    const struct mtxmatrix * A,
    const struct mtxvector * x,
    double beta,
    struct mtxvector * y,
    enum mtxprecision precision,
    enum mtxtransposition trans,
    const char * format,
    int repeat,
    int verbose,
    FILE * diagf,
    bool quiet)
{
    int err;
    struct timespec t0, t1;

    if (precision == mtx_single) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxmatrix_sgemv: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            err = mtxmatrix_sgemv(trans, alpha, A, x, beta, y);
            if (err) {
                if (verbose > 0)
                    fprintf(diagf, "\n");
                return err;
            }
            if (verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(diagf, "%'.6f seconds\n",
                        timespec_duration(t0, t1));
            }
        }
    } else if (precision == mtx_double) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxmatrix_dgemv: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            err = mtxmatrix_dgemv(trans, alpha, A, x, beta, y);
            if (err) {
                if (verbose > 0)
                    fprintf(diagf, "\n");
                return err;
            }
            if (verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(diagf, "%'.6f seconds\n",
                        timespec_duration(t0, t1));
            }
        }
    } else {
        return MTX_ERR_INVALID_PRECISION;
    }

    if (!quiet) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        err = mtxvector_fwrite(
            y, y->type == mtxvector_coordinate ? mtxfile_coordinate : mtxfile_array,
            stdout, format, &bytes_written);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
        }
    }

    return MTX_SUCCESS;
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

    /* 2. Read the matrix from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxmatrix_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxmatrix A;
    int lines_read;
    int64_t bytes_read;
    err = mtxmatrix_read(
        &A, args.precision, args.matrix_type,
        args.A_path ? args.A_path : "", args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d: %s\n",
                program_invocation_short_name,
                args.A_path, lines_read+1,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.A_path, mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    /* 3. Read the vector ‘x’ from a Matrix Market file, or use a
     * vector of all ones. */
    struct mtxvector x;
    if (args.x_path && strlen(args.x_path) > 0) {
        if (args.verbose) {
            fprintf(diagf, "mtxvector_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read;
        int64_t bytes_read;
        err = mtxvector_read(
            &x, args.precision, args.vector_type,
            args.x_path ? args.x_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.x_path, lines_read+1,
                    mtxstrerror(err));
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.x_path, mtxstrerror(err));
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
        }
    } else {
        if (A.type == mtxmatrix_coordinate &&
            A.storage.coordinate.field == mtx_field_pattern)
        {
            err = mtxvector_alloc_array(
                &x, args.vector_field, args.precision,
                A.storage.coordinate.num_columns);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                mtxmatrix_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        } else {
            err = mtxmatrix_alloc_row_vector(&A, &x, args.vector_type);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                mtxmatrix_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        }

        err = mtxvector_set_constant_real_single(&x, 1.0f);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxvector_free(&x);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 4. Read the vector ‘y’ from a Matrix Market file, or use a
     * vector of all zeros. */
    struct mtxvector y;
    if (args.y_path) {
        if (args.verbose) {
            fprintf(diagf, "mtxvector_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int lines_read;
        int64_t bytes_read;
        err = mtxvector_read(
            &y, args.precision, args.vector_type,
            args.y_path ? args.y_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.y_path, lines_read+1,
                    mtxstrerror(err));
            mtxvector_free(&x);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.y_path, mtxstrerror(err));
            mtxvector_free(&x);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
        }
    } else {
        if (A.type == mtxmatrix_coordinate &&
            A.storage.coordinate.field == mtx_field_pattern)
        {
            err = mtxvector_alloc_array(
                &y, args.vector_field, args.precision,
                A.storage.coordinate.num_rows);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                mtxvector_free(&x);
                mtxmatrix_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        } else {
            err = mtxmatrix_alloc_column_vector(&A, &y, args.vector_type);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                mtxvector_free(&x);
                mtxmatrix_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        }

        err = mtxvector_set_constant_real_single(&y, 0.0f);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxvector_free(&y);
            mtxvector_free(&x);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* TODO: Allow the user to select transpose or conjugate
     * transpose matrix-vector multiplication. */
    enum mtxtransposition trans = mtx_notrans;

    /* 5. Compute matrix-vector multiplication. */
    err = gemv(args.alpha, &A, &x, args.beta, &y, args.precision, trans,
               args.format, args.repeat, args.verbose, diagf, args.quiet);
    if (err) {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        mtxvector_free(&y);
        mtxvector_free(&x);
        mtxmatrix_free(&A);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 7. Clean up. */
    mtxvector_free(&y);
    mtxvector_free(&x);
    mtxmatrix_free(&A);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
