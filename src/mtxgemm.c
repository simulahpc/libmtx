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
 * Multiply a general, unsymmetric matrix with another matrix.
 *
 * ‘C := alpha*A*B + C’,
 *
 * where ‘A’ is an M-by-K matrix, ‘B’ is a K-by-N matrix and ‘C’ is an
 * M-by-N matrix, and ‘alpha’ is a scalar constant.
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

const char * program_name = "mtxgemm";
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
    double alpha;
    char * A_path;
    char * B_path;
    char * C_path;
    char * format;
    enum mtxprecision precision;
    enum mtxmatrixtype matrix_type;
    enum mtxvectortype vector_type;
    enum mtxtransposition Atrans;
    enum mtxtransposition Btrans;
    int repeat;
    bool gzip;
    int verbose;
    bool quiet;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->alpha = 1.0;
    args->A_path = NULL;
    args->B_path = NULL;
    args->C_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->matrix_type = mtxbasecoo;
    args->vector_type = mtxbasevector;
    args->Atrans = mtx_notrans;
    args->Btrans = mtx_notrans;
    args->repeat = 1;
    args->gzip = false;
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
    if (args->A_path) free(args->A_path);
    if (args->B_path) free(args->B_path);
    if (args->C_path) free(args->C_path);
    if (args->format) free(args->format);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] [alpha] A [B] [C]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Multiply matrices.\n");
    fprintf(f, "\n");
    fprintf(f, " The operation performed is ‘C := alpha*A*B + C’,\n");
    fprintf(f, " where ‘A’, ‘B’ and ‘C’ are matrices of appropriate dimensions,\n");
    fprintf(f, " and ‘alpha’ is a scalar constant.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  alpha\toptional constant scalar, defaults to 1.0\n");
    fprintf(f, "  A\tpath to Matrix Market file for the matrix A\n");
    fprintf(f, "  B\tOptional path to Matrix Market file for the matrix B.\n");
    fprintf(f, "   \tIf omitted, then an identity matrix is used.\n");
    fprintf(f, "  C\tOptional path to Matrix Market file for the matrix C.\n");
    fprintf(f, "   \tIf omitted, then a zero matrix is used.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’ (default).\n");
    fprintf(f, "  --matrix-type=TYPE\tformat for representing matrices:\n");
    fprintf(f, "\t\t\t‘blas’, ‘dense’, ‘coo’ (default), ‘csr’ or ‘ompcsr’.\n");
    fprintf(f, "  --vector-type=TYPE\ttype of vectors: ‘base’ (default), ‘blas’ or ‘omp’.\n");
    fprintf(f, "  --A-trans=TRANS\t\ttranspose or conjugate transpose A:\n");
    fprintf(f, "\t\t\t‘notrans’, ‘trans’ or ‘conjtrans’. [notrans]\n");
    fprintf(f, "  --B-trans=TRANS\t\ttranspose or conjugate transpose B:\n");
    fprintf(f, "\t\t\t‘notrans’, ‘trans’ or ‘conjtrans’. [notrans]\n");
    fprintf(f, "  --repeat=N\t\trepeat matrix-vector multiplication N times\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
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
    if (err) return err;

    /* Parse program options. */
    int num_positional_arguments_consumed = 0;
    while (*nargs < argc) {
        if (strstr(argv[0], "--precision") == argv[0]) {
            int n = strlen("--precision");
            char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxprecision(&args->precision, s, &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--vector-type") == argv[0]) {
            int n = strlen("--vector-type");
            char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxvectortype(&args->vector_type, s, &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--matrix-type") == argv[0]) {
            int n = strlen("--matrix-type");
            char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxmatrixtype(&args->matrix_type, s, &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--A-trans") == argv[0]) {
            int n = strlen("--A-trans");
            char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxtransposition(&args->Atrans, s, &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--B-trans") == argv[0]) {
            int n = strlen("--B-trans");
            char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxtransposition(&args->Btrans, s, &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--repeat") == argv[0]) {
            int n = strlen("--repeat");
            char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->repeat, s, &s, NULL);
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

        /* positional arguments */
        if (num_positional_arguments_consumed == 0) {
            args->A_path = strdup(argv[0]);
            if (!args->A_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 1) {
            char * argv0 = argv[0];
            argv[0] = args->A_path;
            err = parse_double(&args->alpha, argv[0], NULL, NULL);
            if (err) { argv[0] = argv0; program_options_free(args); return err; }
            argv[0] = argv0;
            if (args->A_path) free(args->A_path);
            args->A_path = strdup(argv[0]);
            if (!args->A_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 2) {
            args->B_path = strdup(argv[0]);
            if (!args->B_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 3) {
            args->C_path = strdup(argv[0]);
            if (!args->C_path) { program_options_free(args); return errno; }
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
 * ‘gemm()’ computes matrix-matrix products and prints the result to
 * standard output.
 */
static int gemm(
    double alpha,
    const struct mtxmatrix * A,
    const struct mtxmatrix * B,
    struct mtxmatrix * C,
    enum mtxprecision precision,
    enum mtxtransposition Atrans,
    enum mtxtransposition Btrans,
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
                fprintf(diagf, "mtxmatrix_sgemm: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxmatrix_sgemm(Atrans, Btrans, alpha, A, B, 1.0f, C, &num_flops);
            if (err) {
                if (verbose > 0) fprintf(diagf, "\n");
                return err;
            }
            if (verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                int64_t num_nonzeros = 0;
                mtxmatrix_num_nonzeros(A, &num_nonzeros);
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s, %'.3f Gnz/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1),
                        1.0e-9 * num_nonzeros / timespec_duration(t0, t1));
            }
        }
    } else if (precision == mtx_double) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxmatrix_dgemm: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxmatrix_dgemm(Atrans, Btrans, alpha, A, B, 1.0, C, &num_flops);
            if (err) {
                if (verbose > 0) fprintf(diagf, "\n");
                return err;
            }
            if (verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                int64_t num_nonzeros = 0;
                mtxmatrix_num_nonzeros(A, &num_nonzeros);
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s, %'.3f Gnz/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1),
                        1.0e-9 * num_nonzeros / timespec_duration(t0, t1));
            }
        }
    } else { return MTX_ERR_INVALID_PRECISION; }

    if (!quiet) {
        if (verbose > 0) {
            fprintf(diagf, "mtxmatrix_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        err = mtxmatrix_fwrite(
            C, 0, NULL, 0, NULL, mtxfile_array, stdout, format, &bytes_written);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
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
 * ‘main()’.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    FILE * diagf = stderr;
    setlocale(LC_ALL, "");

    /* 1. Parse program options. */
    struct program_options args;
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        return EXIT_FAILURE;
    }

    /* 2. Read the matrix from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxmatrix_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxmatrix A;
    int64_t lines_read = 0;
    int64_t bytes_read;
    err = mtxmatrix_read(
        &A, args.precision, args.matrix_type,
        args.A_path ? args.A_path : "", args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.A_path, lines_read+1,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
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

    /* 3. Read the matrix ‘B’ from a Matrix Market file, or use an
     * identity matrix. */
    struct mtxmatrix B;
    if (args.B_path && strlen(args.B_path) > 0) {
        if (args.verbose) {
            fprintf(diagf, "mtxmatrix_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxmatrix_read(
            &B, args.precision, args.matrix_type,
            args.B_path ? args.B_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.B_path, lines_read+1,
                    mtxstrerror(err));
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.B_path, mtxstrerror(err));
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
        /* TODO: Allocate identity matrix */
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(ENOTSUP));
        mtxmatrix_free(&A);
        program_options_free(&args);
        return EXIT_FAILURE;
        
        /* err = mtxmatrix_alloc_row_vector(&A, &x, args.vector_type); */
        /* if (err) { */
        /*     fprintf(stderr, "%s: %s\n", */
        /*             program_invocation_short_name, */
        /*             mtxstrerror(err)); */
        /*     mtxmatrix_free(&A); */
        /*     program_options_free(&args); */
        /*     return EXIT_FAILURE; */
        /* } */

        /* err = mtxvector_set_constant_real_single(&x, 1.0f); */
        /* if (err) { */
        /*     fprintf(stderr, "%s: %s\n", */
        /*             program_invocation_short_name, */
        /*             mtxstrerror(err)); */
        /*     mtxmatrix_free(&B); */
        /*     mtxmatrix_free(&A); */
        /*     program_options_free(&args); */
        /*     return EXIT_FAILURE; */
        /* } */
    }

    /* 4. Read the matrix ‘C’ from a Matrix Market file, or use a
     * matrix of all zeros. */
    struct mtxmatrix C;
    if (args.C_path) {
        if (args.verbose) {
            fprintf(diagf, "mtxmatrix_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxmatrix_read(
            &C, args.precision, args.matrix_type,
            args.C_path ? args.C_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.C_path, lines_read+1,
                    mtxstrerror(err));
            mtxmatrix_free(&B);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.C_path, mtxstrerror(err));
            mtxmatrix_free(&B);
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
        /* TODO: Allocate zero matrix */
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(ENOTSUP));
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
        program_options_free(&args);
        return EXIT_FAILURE;

        /* err = mtxmatrix_alloc_column_vector(&A, &y, args.vector_type); */
        /* if (err) { */
        /*     fprintf(stderr, "%s: %s\n", */
        /*             program_invocation_short_name, */
        /*             mtxstrerror(err)); */
        /*     mtxmatrix_free(&B); */
        /*     mtxmatrix_free(&A); */
        /*     program_options_free(&args); */
        /*     return EXIT_FAILURE; */
        /* } */

        /* err = mtxvector_set_constant_real_single(&y, 0.0f); */
        /* if (err) { */
        /*     fprintf(stderr, "%s: %s\n", */
        /*             program_invocation_short_name, */
        /*             mtxstrerror(err)); */
        /*     mtxmatrix_free(&C); */
        /*     mtxmatrix_free(&B); */
        /*     mtxmatrix_free(&A); */
        /*     program_options_free(&args); */
        /*     return EXIT_FAILURE; */
        /* } */
    }

    /* 5. Compute matrix-matrix multiplication. */
    err = gemm(args.alpha, &A, &B, &C, args.precision, args.Atrans, args.Btrans,
               args.format, args.repeat, args.verbose, diagf, args.quiet);
    if (err) {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        mtxmatrix_free(&C);
        mtxmatrix_free(&B);
        mtxmatrix_free(&A);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 6. Clean up. */
    mtxmatrix_free(&C);
    mtxmatrix_free(&B);
    mtxmatrix_free(&A);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
