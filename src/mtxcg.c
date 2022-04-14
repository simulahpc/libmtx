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
 * Use the conjugate gradient method to (approximately) solve a
 * symmetric, positive definite linear system of equations `Ax=b'.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#include <errno.h>

#include <float.h>
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
    char * b_path;
    char * x_path;
    char * format;
    enum mtxprecision precision;
    bool gzip;
    double atol;
    double rtol;
    int max_iterations;
    int progress;
    int restart;
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
    args->b_path = NULL;
    args->x_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->gzip = false;
    args->atol = 0;
    args->rtol = 1e-6;
    args->max_iterations = 100;
    args->progress = 0;
    args->restart = 0;
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
    if (args->b_path)
        free(args->b_path);
    if (args->x_path)
        free(args->x_path);
    if(args->format)
        free(args->format);
}

/**
 * `program_options_print_usage()` prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A [b] [x]\n", program_name);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Solve a linear system with the conjugate gradient (CG) method.\n");
    fprintf(f, "\n");
    fprintf(f, " The linear system `A*x=b' is (approximately) solved using\n");
    fprintf(f, " the conjugate gradient method. The matrix `A' should be\n");
    fprintf(f, " symmetric and positive definite, `b' is the right-hand side\n");
    fprintf(f, " and `x' is the solution.\n");
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
    fprintf(f, "  --restart=N\t\trestart the iterative solver every N iterations.\n");
    fprintf(f, "  --progress=N\t\tprint a progress report every N iterations.\n");
    fprintf(f, "\t\t\tThe default is to not print any progress report.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: single or double. (default: double)\n");
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

        if (strcmp((*argv)[0], "--restart") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32_ex((*argv)[1], NULL, &args->restart, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--restart=") == (*argv)[0]) {
            err = parse_int32_ex(
                (*argv)[0] + strlen("--restart="), NULL,
                &args->restart, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--progress") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32_ex((*argv)[1], NULL, &args->progress, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--progress=") == (*argv)[0]) {
            err = parse_int32_ex(
                (*argv)[0] + strlen("--progress="), NULL,
                &args->progress, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--atol") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_double_ex((*argv)[1], NULL, &args->atol, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--atol=") == (*argv)[0]) {
            err = parse_double_ex(
                (*argv)[0] + strlen("--atol="), NULL,
                &args->atol, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--rtol") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_double_ex((*argv)[1], NULL, &args->rtol, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--rtol=") == (*argv)[0]) {
            err = parse_double_ex(
                (*argv)[0] + strlen("--rtol="), NULL,
                &args->rtol, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--max-iterations") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32_ex((*argv)[1], NULL, &args->max_iterations, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--max-iterations=") == (*argv)[0]) {
            err = parse_int32_ex(
                (*argv)[0] + strlen("--max-iterations="), NULL,
                &args->max_iterations, NULL);
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
            args->A_path = strdup((*argv)[0]);
            if (!args->A_path) {
                program_options_free(args);
                return errno;
            }
        } else if (num_positional_arguments_consumed == 1) {
            args->b_path = strdup((*argv)[0]);
            if (!args->b_path) {
                program_options_free(args);
                return errno;
            }
        } else if (num_positional_arguments_consumed == 2) {
            args->x_path = strdup((*argv)[0]);
            if (!args->x_path) {
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

    if (num_positional_arguments_consumed < 1) {
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
 * `main()'.
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

    /* 2. Read the matrix from a Matrix Market file. */
    struct mtx A;
    if (args.verbose > 0) {
        fprintf(diagf, "mtx_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int line_number, column_number;
    err = mtx_read(
        &A, args.precision, args.A_path ? args.A_path : "", args.gzip,
        &line_number, &column_number);
    if (err && (line_number == -1 && column_number == -1)) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.A_path, mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d:%d: %s\n",
                program_invocation_short_name,
                args.A_path, line_number, column_number,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 3. Read the vector b from a Matrix Market file, or use a vector
     * of all ones. */
    struct mtx b;
    if (args.b_path && strlen(args.b_path) > 0) {
        if (args.verbose) {
            fprintf(diagf, "mtx_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int line_number, column_number;
        err = mtx_read(
            &b, args.precision, args.b_path, args.gzip,
            &line_number, &column_number);
        if (err && (line_number == -1 && column_number == -1)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.b_path, mtxstrerror(err));
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d:%d: %s\n",
                    program_invocation_short_name,
                    args.b_path, line_number, column_number,
                    mtxstrerror(err));
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    } else {
        err = mtx_alloc_vector_array(
            &b, A.field, args.precision, 0, NULL, A.num_columns);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtx_set_constant_real_single(&b, 1.0f);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 4. Read the vector x from a Matrix Market file, or use a vector
     * of all zeros. */
    struct mtx x;
    if (args.x_path) {
        if (args.verbose) {
            fprintf(diagf, "mtx_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int line_number, column_number;
        err = mtx_read(
            &x, args.precision, args.x_path ? args.x_path : "", args.gzip,
            &line_number, &column_number);
        if (err && (line_number == -1 && column_number == -1)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.x_path, mtxstrerror(err));
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d:%d: %s\n",
                    program_invocation_short_name,
                    args.x_path, line_number, column_number,
                    mtxstrerror(err));
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    } else {
        err = mtx_alloc_vector_array(
            &x, A.field, args.precision, 0, NULL, A.num_rows);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtx_set_zero(&x);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* Allocate working space for the conjugate gradient algorithm. */
    struct mtx_scg_workspace * scg_workspace = NULL;
    struct mtx_dcg_workspace * dcg_workspace = NULL;
    if (A.field == mtx_real) {
        if (args.precision == mtx_single) {
            err = mtx_scg_workspace_alloc(
                &scg_workspace, &A, &b, &x);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                mtx_free(&x);
                mtx_free(&b);
                mtx_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        } else if (args.precision == mtx_double) {
            err = mtx_dcg_workspace_alloc(
                &dcg_workspace, &A, &b, &x);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(err));
                mtx_free(&x);
                mtx_free(&b);
                mtx_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_INVALID_PRECISION));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 5. Solve the linear system using the conjugate gradient
     * method. */
    int num_iterations = 0;
    while (num_iterations < args.max_iterations) {
        int next_progress =
            (args.progress > 0 && args.progress <= args.max_iterations)
            ? (args.progress - (num_iterations % args.progress))
            : args.max_iterations+1;
        int next_restart =
            (args.restart > 0 && args.restart <= args.max_iterations)
            ? args.restart - (num_iterations % args.restart)
            : args.max_iterations+1;
        int max_iterations_in_current_round =
            args.max_iterations - num_iterations;
        if (max_iterations_in_current_round > next_progress)
            max_iterations_in_current_round = next_progress;
        if (max_iterations_in_current_round > next_restart)
            max_iterations_in_current_round = next_restart;
        int num_iterations_in_current_round;

        if (A.field == mtx_real) {
            if (args.precision == mtx_single) {
                if (args.verbose > 0 ||
                    next_progress <= args.max_iterations ||
                    next_restart <= args.max_iterations)
                {
                    fprintf(diagf, "mtx_scg: ");
                    fflush(diagf);
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                }

                float b_nrm2;
                float r_nrm2;
                err = mtx_scg(
                    &A, &x, &b,
                    args.atol, args.rtol, max_iterations_in_current_round,
                    &num_iterations_in_current_round, &b_nrm2, &r_nrm2,
                    scg_workspace);
                if (err != MTX_SUCCESS && err != MTX_ERR_NOT_CONVERGED) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name,
                            mtxstrerror(err));
                    mtx_scg_workspace_free(scg_workspace);
                    mtx_free(&x);
                    mtx_free(&b);
                    mtx_free(&A);
                    program_options_free(&args);
                    return EXIT_FAILURE;
                }
                num_iterations += num_iterations_in_current_round;

                if (args.verbose > 0 ||
                    next_progress <= args.max_iterations ||
                    next_restart <= args.max_iterations)
                {
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    if (err == MTX_SUCCESS) {
                        fprintf(
                            diagf, "%.6f seconds - converged after %d iterations, "
                            "residual 2-norm %.*g, right-hand side 2-norm %.*g, "
                            "relative residual %.*g, "
                            "absolute tolerance %.*g, relative tolerance %.*g\n",
                            timespec_duration(t0, t1),
                            num_iterations, FLT_DIG, r_nrm2,FLT_DIG, b_nrm2,
                            FLT_DIG, fabs(b_nrm2) > FLT_EPSILON ? r_nrm2 / b_nrm2 : INFINITY,
                            DBL_DIG, args.atol, DBL_DIG, args.rtol);
                    } else if (err == MTX_ERR_NOT_CONVERGED) {
                        fprintf(
                            diagf, "%.6f seconds - not converged after %d iterations, "
                            "residual 2-norm %.*g, right-hand side 2-norm %.*g, "
                            "relative residual %.*g, "
                            "absolute tolerance %.*g, relative tolerance %.*g%s\n",
                            timespec_duration(t0, t1),
                            num_iterations, FLT_DIG, r_nrm2, FLT_DIG, b_nrm2,
                            FLT_DIG, fabs(b_nrm2) > FLT_EPSILON ? r_nrm2 / b_nrm2 : INFINITY,
                            DBL_DIG, args.atol, DBL_DIG, args.rtol,
                            next_restart == num_iterations_in_current_round ?
                            ", restarting" : "");
                    }
                }
                if (err == MTX_SUCCESS)
                    break;

                if (num_iterations < args.max_iterations &&
                    next_restart == num_iterations_in_current_round)
                {
                    err = mtx_dcg_restart(dcg_workspace, &A, &b, &x);
                    if (err) {
                        fprintf(stderr, "%s: %s\n",
                                program_invocation_short_name,
                                mtxstrerror(err));
                        mtx_dcg_workspace_free(dcg_workspace);
                        mtx_free(&x);
                        mtx_free(&b);
                        mtx_free(&A);
                        program_options_free(&args);
                        return EXIT_FAILURE;
                    }
                }

            } else if (args.precision == mtx_double) {
                if (args.verbose > 0 ||
                    next_progress <= args.max_iterations ||
                    next_restart <= args.max_iterations)
                {
                    fprintf(diagf, "mtx_dcg: ");
                    fflush(diagf);
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                }

                int num_iterations_in_current_round;
                double b_nrm2;
                double r_nrm2;
                err = mtx_dcg(
                    &A, &x, &b,
                    args.atol, args.rtol, max_iterations_in_current_round,
                    &num_iterations_in_current_round, &b_nrm2, &r_nrm2,
                    dcg_workspace);
                if (err != MTX_SUCCESS && err != MTX_ERR_NOT_CONVERGED) {
                    if (args.verbose > 0)
                        fprintf(diagf, "\n");
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name,
                            mtxstrerror(err));
                    mtx_dcg_workspace_free(dcg_workspace);
                    mtx_free(&x);
                    mtx_free(&b);
                    mtx_free(&A);
                    program_options_free(&args);
                    return EXIT_FAILURE;
                }
                num_iterations += num_iterations_in_current_round;

                if (args.verbose > 0 ||
                    next_progress <= args.max_iterations ||
                    next_restart <= args.max_iterations)
                {
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    if (err == MTX_SUCCESS) {
                        fprintf(
                            diagf, "%.6f seconds - converged after %d iterations, "
                            "residual 2-norm %.*g, right-hand side 2-norm %.*g, "
                            "relative residual %.*g, "
                            "absolute tolerance %.*g, relative tolerance %.*g\n",
                            timespec_duration(t0, t1),
                            num_iterations, DBL_DIG, r_nrm2, DBL_DIG, b_nrm2,
                            DBL_DIG, fabs(b_nrm2) > DBL_EPSILON ? r_nrm2 / b_nrm2 : INFINITY,
                            DBL_DIG, args.atol, DBL_DIG, args.rtol);
                    } else if (err == MTX_ERR_NOT_CONVERGED) {
                        fprintf(
                            diagf, "%.6f seconds - not converged after %d iterations, "
                            "residual 2-norm %.*g, right-hand side 2-norm %.*g, "
                            "relative residual %.*g, "
                            "absolute tolerance %.*g, relative tolerance %.*g%s\n",
                            timespec_duration(t0, t1),
                            num_iterations, DBL_DIG, r_nrm2, DBL_DIG, b_nrm2,
                            DBL_DIG, fabs(b_nrm2) > DBL_EPSILON ? r_nrm2 / b_nrm2 : INFINITY,
                            DBL_DIG, args.atol, DBL_DIG, args.rtol,
                            next_restart == num_iterations_in_current_round ?
                            ", restarting" : "");
                    }
                }
                if (err == MTX_SUCCESS)
                    break;

                if (num_iterations < args.max_iterations
                    && next_restart == num_iterations_in_current_round)
                {
                    err = mtx_dcg_restart(dcg_workspace, &A, &b, &x);
                    if (err) {
                        fprintf(stderr, "%s: %s\n",
                                program_invocation_short_name,
                                mtxstrerror(err));
                        mtx_dcg_workspace_free(dcg_workspace);
                        mtx_free(&x);
                        mtx_free(&b);
                        mtx_free(&A);
                        program_options_free(&args);
                        return EXIT_FAILURE;
                    }
                }

            } else {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxstrerror(MTX_ERR_INVALID_PRECISION));
                mtx_free(&x);
                mtx_free(&b);
                mtx_free(&A);
                program_options_free(&args);
                return EXIT_FAILURE;
            }

        } else if (A.field == mtx_complex ||
                   A.field == mtx_integer ||
                   A.field == mtx_pattern)
        {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    strerror(ENOTSUP));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_INVALID_MTX_FIELD));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    if (A.field == mtx_real) {
        if (args.precision == mtx_single) {
            mtx_scg_workspace_free(scg_workspace);
        } else if (args.precision == mtx_double) {
            mtx_dcg_workspace_free(dcg_workspace);
        } else {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_INVALID_PRECISION));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 6. Write the result vector to standard output. */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtx_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtx_add_comment_line_printf(
            &x, "%% This file was generated by %s %s\n",
            program_name, program_version);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtx_fwrite(&x, stdout, args.format);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtx_free(&x);
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 7. Clean up. */
    mtx_free(&x);
    mtx_free(&b);
    mtx_free(&A);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
