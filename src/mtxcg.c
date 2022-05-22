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
 * Last modified: 2022-05-22
 *
 * Use the conjugate gradient method to (approximately) solve a
 * symmetric, positive definite linear system of equations ‘Ax=b’.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#include <errno.h>

#include <float.h>
#include <locale.h>
#include <math.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxcg";
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
    char * A_path;
    char * b_path;
    char * x_path;
    char * format;
    enum mtxprecision precision;
    enum mtxmatrixtype matrix_type;
    enum mtxvectortype vector_type;
    double atol;
    double rtol;
    int max_iterations;
    int progress;
    int restart;
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
    args->A_path = NULL;
    args->b_path = NULL;
    args->x_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->matrix_type = mtxmatrix_coo;
    args->vector_type = mtxvector_base;
    args->atol = 0;
    args->rtol = 1e-6;
    args->max_iterations = 100;
    args->progress = 0;
    args->restart = 0;
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
    if (args->b_path) free(args->b_path);
    if (args->x_path) free(args->x_path);
    if (args->format) free(args->format);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A [b] [x]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Solve a linear system with the conjugate gradient (CG) method.\n");
    fprintf(f, "\n");
    fprintf(f, " The linear system ‘A*x=b’ is (approximately) solved using\n");
    fprintf(f, " the conjugate gradient method. The matrix ‘A’ should be\n");
    fprintf(f, " symmetric and positive definite, ‘b’ is the right-hand side\n");
    fprintf(f, " and ‘x’ is the solution.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  A\tPath to Matrix Market file for the matrix A.\n");
    fprintf(f, "  b\tOptional path to Matrix Market file for the vector b.\n");
    fprintf(f, "   \tIf omitted, then a vector of ones is used.\n");
    fprintf(f, "  x\tOptional path to Matrix Market file for the vector x.\n");
    fprintf(f, "   \tIf omitted, then a vector of zeros is used.\n");
    fprintf(f, "\n");
    fprintf(f, " Options for the conjugate gradient solver are:\n");
    fprintf(f, "  --atol=TOL\t\tabsolute tolerance, used to declare convergence\n");
    fprintf(f, "\t\t\tif the norm of the residual drops below (default: 0)\n");
    fprintf(f, "  --rtol=TOL\t\trelative tolerance, used to declare convergence\n");
    fprintf(f, "\t\t\tif the norm of the residual divided by the norm of the \n");
    fprintf(f, "\t\t\tright-hand side drops below (default: 1e-6)\n");
    fprintf(f, "  --max-iterations=N\tmaximum number of iterations (default: 100)\n");
    fprintf(f, "  --progress=N\t\tprint a progress report every N iterations.\n");
    fprintf(f, "\t\t\tThe default is to not print any progress report.\n");
    fprintf(f, "  --restart=N\t\trestart the iterative solver every N iterations.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: single or double. (default: double)\n");
    fprintf(f, "  --matrix-type=TYPE\tformat for representing matrices:\n");
    fprintf(f, "\t\t\t‘blas’, ‘dense’, ‘coo’ (default) or ‘csr’.\n");
    fprintf(f, "  --vector-type=TYPE\ttype of vectors: ‘base’ (default), ‘blas’ or ‘omp’.\n");
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
        if (strcmp(argv[0], "--precision") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxprecision_parse(&args->precision, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--precision=") == argv[0]) {
            char * s = argv[0] + strlen("--precision=");
            err = mtxprecision_parse(&args->precision, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--matrix-type") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxmatrixtype_parse(&args->matrix_type, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--matrix-type=") == argv[0]) {
            char * s = argv[0] + strlen("--matrix-type=");
            err = mtxmatrixtype_parse(&args->matrix_type, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--vector-type") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxvectortype_parse(&args->vector_type, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--vector-type=") == argv[0]) {
            char * s = argv[0] + strlen("--vector-type=");
            err = mtxvectortype_parse(&args->vector_type, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--progress") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int32_ex(argv[0], NULL, &args->progress, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--progress=") == argv[0]) {
            err = parse_int32_ex(
                argv[0] + strlen("--progress="), NULL, &args->progress, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--restart") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int32_ex(argv[0], NULL, &args->restart, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--restart=") == argv[0]) {
            err = parse_int32_ex(
                argv[0] + strlen("--restart="), NULL, &args->restart, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--atol") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_double_ex(argv[0], NULL, &args->atol, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--atol=") == argv[0]) {
            err = parse_double_ex(
                argv[0] + strlen("--atol="), NULL, &args->atol, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--rtol") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_double_ex(argv[0], NULL, &args->rtol, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--rtol=") == argv[0]) {
            err = parse_double_ex(
                argv[0] + strlen("--rtol="), NULL, &args->rtol, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--max-iterations") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int32_ex(argv[0], NULL, &args->max_iterations, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--max-iterations=") == argv[0]) {
            err = parse_int32_ex(
                argv[0] + strlen("--max-iterations="), NULL, &args->max_iterations, NULL);
            if (err) { program_options_free(args); return err; }
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
        if (strlen(argv[0]) > 0 && argv[0][0] == '-') {
            program_options_free(args);
            return EINVAL;
        }

        /*
         * Parse positional arguments.
         */
        if (num_positional_arguments_consumed == 0) {
            args->A_path = strdup(argv[0]);
            if (!args->A_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 1) {
            args->b_path = strdup(argv[0]);
            if (!args->b_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 2) {
            args->x_path = strdup(argv[0]);
            if (!args->x_path) { program_options_free(args); return errno; }
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
 * ‘cg()’ solves a linear system using CG.
 */
static int cg(
    const struct mtxmatrix * A,
    const struct mtxvector * b,
    struct mtxvector * x,
    enum mtxprecision precision,
    enum mtxvectortype vectortype,
    double atol,
    double rtol,
    int max_iterations,
    int progress,
    int restart,
    const char * format,
    int verbose,
    FILE * diagf,
    bool quiet)
{
    int err;
    struct timespec t0, t1;

    if (verbose > 0) {
        fprintf(diagf, "mtxcg_init: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* allocate working space for the solver */
    struct mtxcg cg;
    int64_t num_flops = 0;
    err = mtxcg_init(&cg, A, vectortype);
    if (err) {
        if (verbose > 0) fprintf(diagf, "\n");
        return err;
    }

    if (verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    int num_iterations = 0;
    double r0_nrm2;
    bool recompute_residual = false;
    while (num_iterations < max_iterations) {
        int next_progress =
            (progress > 0 && progress <= max_iterations)
            ? (progress - (num_iterations % progress))
            : max_iterations+1;
        int next_restart =
            (restart > 0 && restart <= max_iterations)
            ? restart - (num_iterations % restart)
            : max_iterations+1;
        int max_iterations_in_current_round =
            max_iterations - num_iterations;
        if (max_iterations_in_current_round > next_progress)
            max_iterations_in_current_round = next_progress;
        if (max_iterations_in_current_round > next_restart)
            max_iterations_in_current_round = next_restart;

        if (verbose > 0 ||
            next_progress <= max_iterations ||
            next_restart <= max_iterations)
        {
            fprintf(diagf, "mtxcg_solve: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t num_flops = 0;
        int num_iterations_in_current_round;
        double b_nrm2;
        double r_nrm2;
        err = mtxcg_solve(
            &cg, b, x, atol, rtol, max_iterations_in_current_round,
            recompute_residual, &num_iterations_in_current_round,
            &b_nrm2, &r_nrm2, num_iterations == 0 ? &r0_nrm2 : NULL, &num_flops);
        if (err != MTX_SUCCESS && err != MTX_ERR_NOT_CONVERGED) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxcg_free(&cg);
            return err;
        }
        num_iterations += num_iterations_in_current_round;
        recompute_residual = false;

        if (verbose > 0 ||
            next_progress <= max_iterations ||
            next_restart <= max_iterations)
        {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if (err == MTX_SUCCESS) {
                fprintf(
                    diagf, "%.6f seconds (%'.3f Gflop/s) - "
                    "converged after %d iterations, "
                    "right-hand side 2-norm %.*g, "
                    "residual 2-norm %.*g, "
                    "initial residual 2-norm %.*g, "
                    "relative residual %.*g, "
                    "absolute tolerance %.*g, relative tolerance %.*g\n",
                    timespec_duration(t0, t1),
                    1.0e-9 * num_flops / timespec_duration(t0, t1),
                    num_iterations, DBL_DIG, b_nrm2, DBL_DIG, r_nrm2, DBL_DIG, r0_nrm2,
                    DBL_DIG, fabs(b_nrm2) > DBL_EPSILON ? r_nrm2 / b_nrm2 : INFINITY,
                    DBL_DIG, atol, DBL_DIG, rtol);
            } else if (err == MTX_ERR_NOT_CONVERGED) {
                fprintf(
                    diagf, "%.6f seconds (%'.3f Gflop/s) - "
                    "not converged after %d iterations, "
                    "right-hand side 2-norm %.*g, "
                    "residual 2-norm %.*g, "
                    "initial residual 2-norm %.*g, "
                    "relative residual %.*g, "
                    "absolute tolerance %.*g, relative tolerance %.*g%s\n",
                    timespec_duration(t0, t1),
                    1.0e-9 * num_flops / timespec_duration(t0, t1),
                    num_iterations, DBL_DIG, b_nrm2, DBL_DIG, r_nrm2, DBL_DIG, r0_nrm2,
                    DBL_DIG, fabs(b_nrm2) > DBL_EPSILON ? r_nrm2 / b_nrm2 : INFINITY,
                    DBL_DIG, atol, DBL_DIG, rtol,
                    num_iterations < max_iterations &&
                    next_restart == num_iterations_in_current_round ? ", restarting" : "");
                recompute_residual = true;
            }
        }
        if (err == MTX_SUCCESS) break;
    }
    mtxcg_free(&cg);

    /* write the solution vector to standard output */
    if (!quiet) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        err = mtxvector_fwrite(
            x, 0, NULL, mtxfile_array, stdout, format, &bytes_written);
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
    int64_t bytes_read = 0;
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

    /* 3. Read the vector ‘b’ from a Matrix Market file, or use a
     * vector of all ones. */
    struct mtxvector b;
    if (args.b_path && strlen(args.b_path) > 0) {
        if (args.verbose) {
            fprintf(diagf, "mtxvector_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxvector_read(
            &b, args.precision, args.vector_type,
            args.b_path ? args.b_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.b_path, lines_read+1,
                    mtxstrerror(err));
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.b_path, mtxstrerror(err));
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
        err = mtxmatrix_alloc_row_vector(&A, &b, args.vector_type);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxvector_set_constant_real_single(&b, 1.0f);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxvector_free(&b);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 4. Read the vector ‘x’ from a Matrix Market file, or use a
     * vector of all zeros. */
    struct mtxvector x;
    if (args.x_path) {
        if (args.verbose) {
            fprintf(diagf, "mtxvector_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
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
            mtxvector_free(&b);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.x_path, mtxstrerror(err));
            mtxvector_free(&b);
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
        err = mtxmatrix_alloc_column_vector(&A, &x, args.vector_type);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxvector_free(&b);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxvector_setzero(&x);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxvector_free(&x);
            mtxvector_free(&b);
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 5. Solve the linear system using the conjugate gradient
     * method. */
    err = cg(&A, &b, &x, args.precision, args.vector_type,
             args.atol, args.rtol, args.max_iterations,
             args.progress, args.restart,
             args.format, args.verbose, diagf, args.quiet);
    if (err) {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(err));
        mtxvector_free(&x);
        mtxvector_free(&b);
        mtxmatrix_free(&A);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 6. clean up */
    mtxvector_free(&x);
    mtxvector_free(&b);
    mtxmatrix_free(&A);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
