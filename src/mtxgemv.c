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
 * Last modified: 2022-05-28
 *
 * Multiply a general, unsymmetric matrix with a vector.
 *
 * ‘y := alpha*A*x + y’,
 *
 * where ‘A’ is a matrix, ‘x’ and ‘y’ are vectors and ‘alpha’ is a
 * scalar constant.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

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
    char * x_path;
    char * y_path;
    char * format;
    enum mtxprecision precision;
    enum mtxmatrixtype matrix_type;
    enum mtxvectortype vector_type;
    enum mtxpartitioning partition;
    int64_t blksize;
    enum mtxtransposition trans;
    enum mtxgemvoverlap overlap;
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
    args->x_path = NULL;
    args->y_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->matrix_type = mtxmatrix_coo;
    args->vector_type = mtxbasevector;
    args->partition = mtx_block;
    args->blksize = 1;
    args->trans = mtx_notrans;
    args->overlap = mtxgemvoverlap_none;
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
    if (args->x_path) free(args->x_path);
    if (args->y_path) free(args->y_path);
    if (args->format) free(args->format);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] [alpha] A [x] [y]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Multiply a matrix by a vector.\n");
    fprintf(f, "\n");
    fprintf(f, " The operation performed is ‘y := alpha*A*x + y’,\n");
    fprintf(f, " where ‘A’ is a matrix, ‘x’ and ‘y’ are vectors, and\n");
    fprintf(f, " ‘alpha’ and is a scalar constant.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  alpha\toptional constant scalar, defaults to 1.0\n");
    fprintf(f, "  A\tpath to Matrix Market file for the matrix A\n");
    fprintf(f, "  x\tOptional path to Matrix Market file for the vector x.\n");
    fprintf(f, "   \tIf omitted, then a vector of ones is used.\n");
    fprintf(f, "  y\tOptional path to Matrix Market file for the vector y.\n");
    fprintf(f, "   \tIf omitted, then a vector of zeros is used.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’ (default).\n");
    fprintf(f, "  --matrix-type=TYPE\tformat for representing matrices:\n");
    fprintf(f, "\t\t\t‘blas’, ‘dense’, ‘coo’ (default), ‘csr’ or ‘ompcsr’.\n");
    fprintf(f, "  --vector-type=TYPE\ttype of vectors: ‘base’ (default), ‘blas’ or ‘omp’.\n");
    fprintf(f, "  --partition=TYPE\tmethod of partitioning: ‘block’ (default), ‘block-cyclic’.\n");
    fprintf(f, "  --blksize=N\t\tblock size to use for block-cyclic partitioning\n");
    fprintf(f, "  --trans=TRANS\t\tCompute transpose or conjugate transpose matrix-vector product.\n");
    fprintf(f, "\t\t\tOptions are ‘notrans’ (default), ‘trans’ or ‘conjtrans’.\n");
    fprintf(f, "  --overlap=TYPE\t\ttype of communication-computation overlap:\n");
    fprintf(f, "\t\t\t‘none’ (default).\n");
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

        if (strcmp(argv[0], "--trans") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxtransposition_parse(&args->trans, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--trans=") == argv[0]) {
            char * s = argv[0] + strlen("--trans=");
            err = mtxtransposition_parse(&args->trans, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--overlap") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxgemvoverlap_parse(&args->overlap, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--overlap=") == argv[0]) {
            char * s = argv[0] + strlen("--overlap=");
            err = mtxgemvoverlap_parse(&args->overlap, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--partition") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxpartitioning_parse(&args->partition, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--partition=") == argv[0]) {
            char * s = argv[0] + strlen("--partition=");
            err = mtxpartitioning_parse(&args->partition, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--blksize") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int64_ex(argv[0], NULL, &args->blksize, NULL);
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--blksize=") == argv[0]) {
            char * s = argv[0] + strlen("--blksize=");
            err = parse_int64_ex(s, NULL, &args->blksize, NULL);
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--repeat") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int32_ex(argv[0], NULL, &args->repeat, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--repeat=") == argv[0]) {
            err = parse_int32_ex(
                argv[0] + strlen("--repeat="), NULL, &args->repeat, NULL);
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
            if (rank == 0) program_options_print_help(stdout);
            MPI_Finalize();
#else
            program_options_print_help(stdout);
#endif
            exit(EXIT_SUCCESS);
        }

        /* If requested, print program version information. */
        if (strcmp(argv[0], "--version") == 0) {
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
            if (rank == 0) program_options_print_version(stdout);
            MPI_Finalize();
#else
            program_options_print_version(stdout);
#endif
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
            args->A_path = strdup(argv[0]);
            if (!args->A_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 1) {
            char * A_path = strdup(argv[0]);
            if (!A_path) { program_options_free(args); return errno; }
            argv[0] = strdup(args->A_path);
            err = parse_double_ex(argv[0], NULL, &args->alpha, NULL);
            if (err) { program_options_free(args); return err; }
            args->A_path = A_path;
        } else if (num_positional_arguments_consumed == 2) {
            args->x_path = strdup(argv[0]);
            if (!args->x_path) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 3) {
            args->y_path = strdup(argv[0]);
            if (!args->y_path) { program_options_free(args); return errno; }
        } else { program_options_free(args); return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
    }

    if (num_positional_arguments_consumed < 1) {
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
        if (rank == 0) program_options_print_usage(stdout);
        MPI_Finalize();
#else
        program_options_print_usage(stdout);
#endif
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

#ifdef LIBMTX_HAVE_MPI
/**
 * ‘distgemv()’ converts Matrix Market files to matrices and vectors
 * of the specified type, computes matrix-vector products and prints
 * the result to standard output.
 */
static int distgemv(
    double alpha,
    const struct mtxdistfile * mtxdistfileA,
    const struct mtxdistfile * mtxdistfilex,
    const struct mtxdistfile * mtxdistfiley,
    enum mtxmatrixtype matrixtype,
    enum mtxvectortype vectortype,
    enum mtxtransposition trans,
    enum mtxgemvoverlap overlap,
    int repeat,
    const char * format,
    enum mtxfileformat mtxfmt,
    int verbose,
    FILE * diagf,
    bool quiet,
    MPI_Comm comm,
    int commsize,
    int rank,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    struct timespec t0, t1;
    enum mtxprecision precision = mtxdistfileA->precision;

    /* 1. convert Matrix Market files to matrix and vectors. */
    if (verbose > 0) {
        fprintf(diagf, "mtxmatrix_dist_from_mtxdistfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    struct mtxmatrix_dist A;
    err = mtxmatrix_dist_from_mtxdistfile(
        &A, mtxdistfileA, matrixtype, comm, disterr);
    if (err) {
        if (verbose > 0) fprintf(diagf, "\n");
        return err;
    }
    if (verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n",
                timespec_duration(t0, t1));
    }

    struct mtxvector_dist x;
    if (mtxdistfilex) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_dist_from_mtxdistfile: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        err = mtxvector_dist_from_mtxdistfile(
            &x, mtxdistfilex, vectortype, comm, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxmatrix_dist_free(&A);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    } else {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_dist_alloc: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* partition the matrix columns into equal-sized blocks */
        int64_t num_columns = A.num_columns;
        int64_t colpartsize = num_columns / commsize +
            (rank < (num_columns % commsize) ? 1 : 0);
        int64_t colpartoffset = rank * (num_columns / commsize) +
            (rank < (num_columns % commsize) ? rank : (num_columns % commsize));

        enum mtxfield field;
        err = mtxmatrix_dist_field(&A, &field);
        enum mtxprecision precision;
        err = err ? err : mtxmatrix_dist_precision(&A, &precision);
        if (mtxdisterror_allreduce(disterr, err)) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxmatrix_dist_free(&A);
            return err;
        }

        int64_t * idx = malloc(colpartsize * sizeof(int64_t));
        err = !idx ? MTX_ERR_ERRNO : MTX_SUCCESS;
        if (mtxdisterror_allreduce(disterr, err)) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxmatrix_dist_free(&A);
            return err;
        }
        for (int64_t i = 0; i < colpartsize; i++)
            idx[i] = colpartoffset+i;

        err = mtxvector_dist_alloc(
            &x, vectortype, field, precision, num_columns,
            colpartsize, idx, comm, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            free(idx);
            mtxmatrix_dist_free(&A);
            return err;
        }
        free(idx);

        err = mtxvector_dist_set_constant_real_single(&x, 1.0f, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxvector_dist_free(&x);
            mtxmatrix_dist_free(&A);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
        }
    }

    struct mtxvector_dist y;
    if (mtxdistfiley) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_dist_from_mtxdistfile: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        err = mtxvector_dist_from_mtxdistfile(
            &y, mtxdistfiley, vectortype, comm, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxvector_dist_free(&x);
            mtxmatrix_dist_free(&A);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
        }
    } else {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_dist_alloc_column_vector: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        err = mtxmatrix_dist_alloc_column_vector(&A, &y, vectortype, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxmatrix_dist_free(&A);
            return err;
        }
        err = mtxvector_dist_setzero(&y, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxvector_dist_free(&y);
            mtxvector_dist_free(&x);
            mtxmatrix_dist_free(&A);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
        }
    }

    /* 2. prepare for matrix-vector multiplication */
    if (verbose > 0) {
        fprintf(diagf, "mtxmatrix_dist_gemv_init: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    struct mtxmatrix_dist_gemv gemv;
    err = mtxmatrix_dist_gemv_init(
        &gemv, trans, &A, &x, &y, overlap, disterr);
    if (err) {
        if (verbose > 0) fprintf(diagf, "\n");
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
        return err;
    }
    if (verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    /* 3. perform matrix-vector multiplication */
    if (precision == mtx_single) {
        for (int i = 0; i < repeat; i++) {
            MPI_Barrier(comm);
            if (verbose > 0) {
                fprintf(diagf, "mtxmatrix_dist_gemv_sgemv: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxmatrix_dist_gemv_sgemv(&gemv, alpha, 1, disterr);
            if (err) {
                if (verbose > 0) fprintf(diagf, "\n");
                mtxmatrix_dist_gemv_free(&gemv);
                mtxvector_dist_free(&y);
                mtxvector_dist_free(&x);
                mtxmatrix_dist_free(&A);
                return err;
            }
            err = mtxmatrix_dist_gemv_wait(&gemv, &num_flops, disterr);
            if (err) {
                if (verbose > 0) fprintf(diagf, "\n");
                mtxmatrix_dist_gemv_free(&gemv);
                mtxvector_dist_free(&y);
                mtxvector_dist_free(&x);
                mtxmatrix_dist_free(&A);
                return err;
            }
            MPI_Barrier(comm);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            err = MPI_Reduce(
                rank == root ? MPI_IN_PLACE : &num_flops,
                &num_flops, 1, MPI_INT64_T, MPI_SUM, root, comm);
            if (err) {
                char mpierrstr[MPI_MAX_ERROR_STRING];
                int mpierrstrlen;
                MPI_Error_string(err, mpierrstr, &mpierrstrlen);
                fprintf(stderr, "%s: MPI_Reduce failed with %s\n",
                        program_invocation_short_name, mpierrstr);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (verbose > 0) {
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s, %'.3f Gnz/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1),
                        1.0e-9 * A.num_nonzeros / timespec_duration(t0, t1));
            }
        }
    } else if (precision == mtx_double) {
        for (int i = 0; i < repeat; i++) {
            MPI_Barrier(comm);
            if (verbose > 0) {
                fprintf(diagf, "mtxmatrix_dist_gemv_dgemv: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxmatrix_dist_gemv_dgemv(&gemv, alpha, 1, disterr);
            if (err) {
                if (verbose > 0) fprintf(diagf, "\n");
                mtxmatrix_dist_gemv_free(&gemv);
                mtxvector_dist_free(&y);
                mtxvector_dist_free(&x);
                mtxmatrix_dist_free(&A);
                return err;
            }
            err = mtxmatrix_dist_gemv_wait(&gemv, &num_flops, disterr);
            if (err) {
                if (verbose > 0) fprintf(diagf, "\n");
                mtxmatrix_dist_gemv_free(&gemv);
                mtxvector_dist_free(&y);
                mtxvector_dist_free(&x);
                mtxmatrix_dist_free(&A);
                return err;
            }
            MPI_Barrier(comm);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            err = MPI_Reduce(
                rank == root ? MPI_IN_PLACE : &num_flops,
                &num_flops, 1, MPI_INT64_T, MPI_SUM, root, comm);
            if (err) {
                char mpierrstr[MPI_MAX_ERROR_STRING];
                int mpierrstrlen;
                MPI_Error_string(err, mpierrstr, &mpierrstrlen);
                fprintf(stderr, "%s: MPI_Reduce failed with %s\n",
                        program_invocation_short_name, mpierrstr);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (verbose > 0) {
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s, %'.3f Gnz/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1),
                        1.0e-9 * A.num_nonzeros / timespec_duration(t0, t1));
            }
        }
    } else {
        mtxmatrix_dist_gemv_free(&gemv);
        mtxvector_dist_free(&y);
        mtxvector_dist_free(&x);
        mtxmatrix_dist_free(&A);
        return MTX_ERR_INVALID_PRECISION;
    }

    /* 4. output results */
    if (!quiet) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_dist_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        err = mtxvector_dist_fwrite(
            &y, mtxfmt, stdout, format, &bytes_written, root, disterr);
        if (err) {
            if (verbose > 0) fprintf(diagf, "\n");
            mtxmatrix_dist_gemv_free(&gemv);
            mtxvector_dist_free(&y);
            mtxvector_dist_free(&x);
            mtxmatrix_dist_free(&A);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
        }
    }

    mtxmatrix_dist_gemv_free(&gemv);
    mtxvector_dist_free(&y);
    mtxvector_dist_free(&x);
    mtxmatrix_dist_free(&A);
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

    /* Set program invocation name. */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);

    /* 1. Initialise MPI. */
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

    /* 2. Parse program options. */
    struct program_options args;
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        if (rank == root) {
            fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                    strerror(err), argv[nargs]);
        }
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank != root)
        args.verbose = false;

    /* 3. Read the ‘A’ matrix from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfileA;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    if (rank == root) {
        err = mtxfile_read(
            &mtxfileA, args.precision,
            args.A_path ? args.A_path : "", args.gzip,
            &lines_read, &bytes_read);
    }
    if (mtxdisterror_allreduce(&disterr, err)) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        if (rank == root && lines_read >= 0) {
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.A_path, lines_read+1,
                    mtxstrerror(err));
        } else if (rank == root) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.A_path, mtxstrerror(err));
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

    if (args.verbose > 0) {
        fprintf(diagf, "mtxdistfile_from_mtxfile_rowwise: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* partition the matrix rows into equal-sized blocks */
    int64_t rowpartsize = 0;
    if (args.partition == mtx_block) {
        if (rank == root) {
            rowpartsize = mtxfileA.size.num_rows / comm_size +
                (rank < (mtxfileA.size.num_rows % comm_size) ? 1 : 0);
        }
        disterr.mpierrcode = MPI_Bcast(&rowpartsize, 1, MPI_INT64_T, root, comm);
        err = disterr.mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name, mtxstrerror(err));
            }
            if (rank == root) mtxfile_free(&mtxfileA);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    struct mtxdistfile mtxdistfileA;
    const int * parts = NULL;
    err = mtxdistfile_from_mtxfile_rowwise(
        &mtxdistfileA, &mtxfileA, args.partition, rowpartsize, args.blksize, parts,
        comm, root, &disterr);
    if (err) {
        if (args.verbose > 0) fprintf(diagf, "\n");
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        }
        if (rank == root) mtxfile_free(&mtxfileA);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank == root) mtxfile_free(&mtxfileA);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 4. Read the ‘x’ vector from a Matrix Market file. */
    struct mtxdistfile mtxdistfilex;
    if (args.x_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile mtxfilex;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        if (rank == root) {
            err = mtxfile_read(
                &mtxfilex, args.precision,
                args.x_path ? args.x_path : "", args.gzip,
                &lines_read, &bytes_read);
        }
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (rank == root && lines_read >= 0) {
                fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                        program_invocation_short_name,
                        args.x_path, lines_read+1,
                        mtxstrerror(err));
            } else if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.x_path, mtxstrerror(err));
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

        if (args.verbose > 0) {
            fprintf(diagf, "mtxdistfile_from_mtxfile_rowwise: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* partition the rows into equal-sized blocks */
        int64_t partsize = 0;
        if (args.partition == mtx_block) {
            if (rank == root) {
                partsize = mtxfilex.size.num_rows / comm_size +
                    (rank < (mtxfilex.size.num_rows % comm_size) ? 1 : 0);
            }
            disterr.mpierrcode = MPI_Bcast(&partsize, 1, MPI_INT64_T, root, comm);
            err = disterr.mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(&disterr, err)) {
                if (args.verbose > 0) fprintf(diagf, "\n");
                if (rank == root) {
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name, mtxstrerror(err));
                }
                if (rank == root) mtxfile_free(&mtxfilex);
                mtxdistfile_free(&mtxdistfileA);
                program_options_free(&args);
                mtxdisterror_free(&disterr);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }

        const int * parts = NULL;
        err = mtxdistfile_from_mtxfile_rowwise(
            &mtxdistfilex, &mtxfilex, args.partition, partsize, args.blksize, parts,
            comm, root, &disterr);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            if (rank == root) mtxfile_free(&mtxfilex);
            mtxdistfile_free(&mtxdistfileA);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        if (rank == root) mtxfile_free(&mtxfilex);

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    }

    /* 5. Read the ‘y’ vector from a Matrix Market file, or use a
     * vector of zeros. */
    struct mtxdistfile mtxdistfiley;
    if (args.y_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        struct mtxfile mtxfiley;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        if (rank == root) {
            err = mtxfile_read(
                &mtxfiley, args.precision,
                args.y_path ? args.y_path : "", args.gzip,
                &lines_read, &bytes_read);
        }
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (rank == root && lines_read >= 0) {
                fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                        program_invocation_short_name,
                        args.y_path, lines_read+1,
                        mtxstrerror(err));
            } else if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.y_path, mtxstrerror(err));
            }
            if (args.x_path) mtxdistfile_free(&mtxdistfilex);
            mtxdistfile_free(&mtxdistfileA);
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

        if (args.verbose > 0) {
            fprintf(diagf, "mtxdistfile_from_mtxfile_rowwise: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* partition the rows into equal-sized blocks */
        int64_t partsize = 0;
        if (args.partition == mtx_block) {
            if (rank == root) {
                partsize = mtxfiley.size.num_rows / comm_size +
                    (rank < (mtxfiley.size.num_rows % comm_size) ? 1 : 0);
            }
            disterr.mpierrcode = MPI_Bcast(&partsize, 1, MPI_INT64_T, root, comm);
            err = disterr.mpierrcode ? MTX_ERR_MPI : MTX_SUCCESS;
            if (mtxdisterror_allreduce(&disterr, err)) {
                if (args.verbose > 0) fprintf(diagf, "\n");
                if (rank == root) {
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name, mtxstrerror(err));
                }
                if (rank == root) mtxfile_free(&mtxfiley);
                if (args.x_path) mtxdistfile_free(&mtxdistfilex);
                mtxdistfile_free(&mtxdistfileA);
                program_options_free(&args);
                mtxdisterror_free(&disterr);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }

        const int * parts = NULL;
        err = mtxdistfile_from_mtxfile_rowwise(
            &mtxdistfiley, &mtxfiley, args.partition, partsize, args.blksize, parts,
            comm, root, &disterr);
        if (err) {
            if (args.verbose > 0) fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            if (rank == root) mtxfile_free(&mtxfiley);
            if (args.x_path) mtxdistfile_free(&mtxdistfilex);
            mtxdistfile_free(&mtxdistfileA);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        if (rank == root) mtxfile_free(&mtxfiley);

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    }

    /* 6. perform matrix-vector multiplication */
    err = distgemv(
        args.alpha, &mtxdistfileA,
        args.x_path ? &mtxdistfilex : NULL,
        args.y_path ? &mtxdistfiley : NULL,
        args.matrix_type, args.vector_type, args.trans, args.overlap, args.repeat,
        args.format, args.y_path ? mtxdistfiley.header.format : mtxfile_array,
        args.verbose, diagf, args.quiet,
        comm, comm_size, rank, root, &disterr);
    if (err) {
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        }
        if (args.y_path) mtxdistfile_free(&mtxdistfiley);
        if (args.x_path) mtxdistfile_free(&mtxdistfilex);
        mtxdistfile_free(&mtxdistfileA);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 6. Clean up. */
    if (args.y_path) mtxdistfile_free(&mtxdistfiley);
    if (args.x_path) mtxdistfile_free(&mtxdistfilex);
    mtxdistfile_free(&mtxdistfileA);
    program_options_free(&args);
    mtxdisterror_free(&disterr);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
#else
/**
 * ‘gemv()’ computes matrix-vector products and prints the result to
 * standard output.
 */
static int gemv(
    double alpha,
    const struct mtxmatrix * A,
    const struct mtxvector * x,
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
            int64_t num_flops = 0;
            err = mtxmatrix_sgemv(trans, alpha, A, x, 1.0f, y, &num_flops);
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
                fprintf(diagf, "mtxmatrix_dgemv: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxmatrix_dgemv(trans, alpha, A, x, 1.0, y, &num_flops);
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
            y, 0, NULL, mtxfile_array, stdout, format, &bytes_written);
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
    int64_t bytes_read;
    err = mtxmatrix_read(
        &A, args.precision, args.matrix_type,
        args.A_path ? args.A_path : "", args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
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

        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxvector_read(
            &x, args.precision, args.vector_type,
            args.x_path ? args.x_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
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
        err = mtxmatrix_alloc_row_vector(&A, &x, args.vector_type);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxmatrix_free(&A);
            program_options_free(&args);
            return EXIT_FAILURE;
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
        int64_t lines_read = 0;
        int64_t bytes_read;
        err = mtxvector_read(
            &y, args.precision, args.vector_type,
            args.y_path ? args.y_path : "", args.gzip,
            &lines_read, &bytes_read);
        if (err && lines_read >= 0) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
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

    /* 5. Compute matrix-vector multiplication. */
    err = gemv(args.alpha, &A, &x, &y, args.precision, args.trans,
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

    /* 6. Clean up. */
    mtxvector_free(&y);
    mtxvector_free(&x);
    mtxmatrix_free(&A);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
#endif
