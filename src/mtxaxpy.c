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
 * Last modified: 2022-01-19
 *
 * Multiply a vector by a scalar and add it to another vector.
 *
 * ‘y := alpha*x + y’,
 *
 * where ‘x’ and ‘y’ are vectors and ‘alpha’ is a scalar constant.
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

const char * program_name = "mtxaxpy";
const char * program_version = "0.1.0";
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
    double alpha;
    char * x_path;
    char * y_path;
    char * format;
    enum mtxprecision precision;
    enum mtxvectortype vector_type;
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
    args->x_path = NULL;
    args->y_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->vector_type = mtxvector_auto;
    args->repeat = 1;
    args->gzip = false;
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
    fprintf(f, "Usage: %s [OPTION..] [alpha] x [y]\n", program_name);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Multiply a vector by a scalar and add it to another vector.\n");
    fprintf(f, "\n");
    fprintf(f, " The operation performed is ‘y := alpha*x + y’, where\n");
    fprintf(f, " where ‘x’ and ‘y’ are vectors and ‘alpha’ is a scalar constant.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  alpha\tOptional constant scalar, defaults to 1.0.\n");
    fprintf(f, "  x\tPath to Matrix Market file for the vector x.\n");
    fprintf(f, "  y\tOptional path to Matrix Market file for the vector y.\n");
    fprintf(f, "   \tIf omitted, then a vector of ones is used.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: ‘single’ or ‘double’. (default: ‘double’)\n");
    fprintf(f, "  --vector-type=TYPE\tformat for representing vectors:\n");
    fprintf(f, "\t\t\t‘auto’, ‘array’ or ‘coordinate’. (default: ‘auto’)\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  --repeat=N\t\trepeat the calculation N times\n");
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
    fprintf(f, "%s %s (Libmtx %s)\n", program_name, program_version,
            libmtx_version);
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * `parse_program_options()` parses program options.
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
        if (strcmp(argv[0], "--precision") == 0) {
            if (argc - *nargs < 2) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            char * s = argv[0];
            err = mtxprecision_parse(&args->precision, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            continue;
        } else if (strstr(argv[0], "--precision=") == argv[0]) {
            char * s = argv[0] + strlen("--precision=");
            err = mtxprecision_parse(&args->precision, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            continue;
        }

        if (strcmp(argv[0], "--vector-type") == 0) {
            if (argc - *nargs < 2) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            char * s = argv[0];
            err = mtxvectortype_parse(
                &args->vector_type, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            continue;
        } else if (strstr(argv[0], "--vector-type=") == argv[0]) {
            char * s = argv[0] + strlen("--vector-type=");
            err = mtxvectortype_parse(
                &args->vector_type, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            continue;
        }

        if (strcmp(argv[0], "-z") == 0 ||
            strcmp(argv[0], "--gzip") == 0 ||
            strcmp(argv[0], "--gunzip") == 0 ||
            strcmp(argv[0], "--ungzip") == 0)
        {
            args->gzip = true;
            (*nargs)++; argv++;
            continue;
        }

        if (strcmp(argv[0], "--format") == 0) {
            if (argc - *nargs < 2) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            args->format = strdup(argv[0]);
            if (!args->format) {
                program_options_free(args);
                return errno;
            }
            (*nargs)++; argv++;
            continue;
        } else if (strstr(argv[0], "--format=") == argv[0]) {
            args->format = strdup(argv[0] + strlen("--format="));
            if (!args->format) {
                program_options_free(args);
                return errno;
            }
            (*nargs)++; argv++;
            continue;
        }

        if (strcmp(argv[0], "--repeat") == 0) {
            if (argc - *nargs < 2) {
                program_options_free(args);
                return EINVAL;
            }
            (*nargs)++; argv++;
            err = parse_int32(argv[0], NULL, &args->repeat, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            (*nargs)++; argv++;
            continue;
        } else if (strstr(argv[0], "--repeat=") == argv[0]) {
            err = parse_int32(
                argv[0] + strlen("--repeat="), NULL,
                &args->repeat, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            (*nargs)++; argv++;
            continue;
        }

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = true;
            (*nargs)++; argv++;
            continue;
        }

        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++;
            continue;
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
            if (rank == 0)
                program_options_print_help(stdout);
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
            if (rank == 0)
                program_options_print_version(stdout);
            MPI_Finalize();
#else
            program_options_print_version(stdout);
#endif
            exit(EXIT_SUCCESS);
        }

        /* Stop parsing options after '--'.  */
        if (strcmp(argv[0], "--") == 0) {
            argc--; argv++;
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
            args->x_path = strdup(argv[0]);
            if (!args->x_path) {
                program_options_free(args);
                return errno;
            }
        } else if (num_positional_arguments_consumed == 1) {
            char * x_path = strdup(argv[0]);
            if (!x_path) {
                program_options_free(args);
                return errno;
            }
            argv[0] = strdup(args->x_path);
            err = parse_double(argv[0], NULL, &args->alpha, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            args->x_path = x_path;
        } else if (num_positional_arguments_consumed == 2) {
            args->y_path = strdup(argv[0]);
            if (!args->y_path) {
                program_options_free(args);
                return errno;
            }
        } else {
            program_options_free(args);
            return EINVAL;
        }

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
        if (rank == 0)
            program_options_print_usage(stdout);
        MPI_Finalize();
#else
        program_options_print_usage(stdout);
#endif
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

#ifdef LIBMTX_HAVE_MPI
/**
 * `distvector_axpy()' converts Matrix Market files to vectors of the
 * given type, adds two vectors and prints the result to standard
 * output.
 */
static int distvector_axpy(
    double alpha,
    struct mtxdistfile * mtxdistfilex,
    struct mtxdistfile * mtxdistfiley,
    enum mtxvectortype vector_type,
    const char * format,
    int repeat,
    int verbose,
    FILE * diagf,
    bool quiet,
    MPI_Comm comm,
    int comm_size,
    int rank,
    int root,
    struct mtxdisterror * disterr)
{
    int err;
    struct timespec t0, t1;
    enum mtxprecision precision = mtxdistfilex->precision;

    /* 1. Convert Matrix Market files to vectors. */
    if (verbose > 0) {
        fprintf(diagf, "mtxdistvector_from_mtxdistfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    struct mtxdistvector x;
    err = mtxdistvector_from_mtxdistfile(
        &x, mtxdistfilex, vector_type, NULL, comm, disterr);
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

    if (verbose > 0) {
        fprintf(diagf, "mtxdistvector_from_mtxdistfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    struct mtxdistvector y;
    err = mtxdistvector_from_mtxdistfile(
        &y, mtxdistfiley, vector_type, NULL, comm, disterr);
    if (err) {
        if (verbose > 0)
            fprintf(diagf, "\n");
        mtxdistvector_free(&x);
        return err;
    }
    if (verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 2. Add the vectors. */
    if (precision == mtx_single) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxdistvector_saxpy: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxdistvector_saxpy(alpha, &x, &y, &num_flops, disterr);
            if (err) {
                if (verbose > 0)
                    fprintf(diagf, "\n");
                mtxdistvector_free(&y);
                mtxdistvector_free(&x);
                return err;
            }
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
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1));
            }
        }
    } else if (precision == mtx_double) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxdistvector_daxpy: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxdistvector_daxpy(alpha, &x, &y, &num_flops, disterr);
            if (err) {
                if (verbose > 0)
                    fprintf(diagf, "\n");
                mtxdistvector_free(&y);
                mtxdistvector_free(&x);
                return err;
            }
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
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1));
            }
        }
    } else {
        mtxdistvector_free(&y);
        mtxdistvector_free(&x);
        return MTX_ERR_INVALID_PRECISION;
    }

    if (!quiet) {
        if (verbose > 0) {
            fprintf(diagf, "mtxdistvector_fwrite_shared: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        enum mtxfileformat mtxfmt =
            y.interior.type == mtxvector_coordinate
            ? mtxfile_coordinate : mtxfile_array;
        err = mtxdistvector_fwrite_shared(
            &y, mtxfmt, stdout, format, &bytes_written, root, disterr);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
            mtxdistvector_free(&x);
            return MTX_ERR_MPI_COLLECTIVE;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
        }
    }

    mtxdistvector_free(&y);
    mtxdistvector_free(&x);
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

    /* 3. Read the ‘x’ vector from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxdistfile_read_shared: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* TODO: Make row partitioning configurable, see mtxpartition.c */
    const enum mtxpartitioning row_partition_type = mtx_block;

    struct mtxdistfile mtxdistfilex;
    int lines_read;
    int64_t bytes_read;
    err = mtxdistfile_read_shared(
        &mtxdistfilex, args.precision,
        args.x_path ? args.x_path : "", args.gzip,
        &lines_read, &bytes_read,
        comm, root, &disterr);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        if (rank == root && lines_read >= 0) {
            fprintf(stderr, "%s: %s:%d: %s\n",
                    program_invocation_short_name,
                    args.x_path, lines_read+1,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
        } else if (rank == root) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.x_path,
                    err == MTX_ERR_MPI_COLLECTIVE
                    ? mtxdisterror_description(&disterr)
                    : mtxstrerror(err));
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

    /* 4. Read the ‘y’ vector from a Matrix Market file, or use a
     * vector of all zeros. */
    struct mtxdistfile mtxdistfiley;
    if (args.y_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxdistfile_read_shared: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read;
        int64_t bytes_read;
        err = mtxdistfile_read_shared(
            &mtxdistfiley, args.precision,
            args.y_path, args.gzip,
            &lines_read, &bytes_read,
            comm, root, &disterr);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root && lines_read >= 0) {
                fprintf(stderr, "%s: %s:%d: %s\n",
                        program_invocation_short_name,
                        args.y_path, lines_read+1,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            } else if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.y_path,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            mtxdistfile_free(&mtxdistfilex);
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
    } else {
        err = mtxdistfile_alloc_copy(&mtxdistfiley, &mtxdistfilex, &disterr);
        if (err) {
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            mtxdistfile_free(&mtxdistfilex);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        err = mtxdistfile_set_constant_real_single(&mtxdistfiley, 0.0f, &disterr);
        if (err) {
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            mtxdistfile_free(&mtxdistfiley);
            mtxdistfile_free(&mtxdistfilex);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    /* 5. Add the vectors. */
    if (mtxdistfilex.header.object == mtxfile_matrix) {
        /* TODO: Add matrices. */
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    strerror(ENOTSUP));
        }
        mtxdistfile_free(&mtxdistfiley);
        mtxdistfile_free(&mtxdistfilex);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;

    } else if (mtxdistfilex.header.object == mtxfile_vector) {
        err = distvector_axpy(
            args.alpha, &mtxdistfilex, &mtxdistfiley, args.vector_type, args.format,
            args.repeat, args.verbose, diagf, args.quiet,
            comm, comm_size, rank, root, &disterr);
        if (err) {
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            mtxdistfile_free(&mtxdistfiley);
            mtxdistfile_free(&mtxdistfilex);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

    } else {
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(MTX_ERR_INVALID_MTX_OBJECT));
        }
        mtxdistfile_free(&mtxdistfiley);
        mtxdistfile_free(&mtxdistfilex);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 5. Clean up. */
    mtxdistfile_free(&mtxdistfiley);
    mtxdistfile_free(&mtxdistfilex);
    program_options_free(&args);
    mtxdisterror_free(&disterr);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
#else
/**
 * `vector_axpy()' converts Matrix Market files to vectors of the
 * given type, adds two vectors and prints the result to standard
 * output.
 */
static int vector_axpy(
    double alpha,
    struct mtxfile * mtxfilex,
    struct mtxfile * mtxfiley,
    enum mtxvectortype vector_type,
    const char * format,
    int repeat,
    int verbose,
    FILE * diagf,
    bool quiet)
{
    int err;
    struct timespec t0, t1;
    enum mtxprecision precision = mtxfilex->precision;

    /* 1. Convert Matrix Market files to vectors. */
    if (verbose > 0) {
        fprintf(diagf, "mtxvector_from_mtxfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    struct mtxvector x;
    err = mtxvector_from_mtxfile(&x, mtxfilex, vector_type);
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

    if (verbose > 0) {
        fprintf(diagf, "mtxvector_from_mtxfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    struct mtxvector y;
    err = mtxvector_from_mtxfile(&y, mtxfiley, vector_type);
    if (err) {
        if (verbose > 0)
            fprintf(diagf, "\n");
        mtxvector_free(&x);
        return err;
    }
    if (verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 2. Add the vectors. */
    if (precision == mtx_single) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxvector_saxpy: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxvector_saxpy(alpha, &x, &y, &num_flops);
            if (err) {
                if (verbose > 0)
                    fprintf(diagf, "\n");
                mtxvector_free(&y);
                mtxvector_free(&x);
                return err;
            }
            if (verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1));
            }
        }
    } else if (precision == mtx_double) {
        for (int i = 0; i < repeat; i++) {
            if (verbose > 0) {
                fprintf(diagf, "mtxvector_daxpy: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }
            int64_t num_flops = 0;
            err = mtxvector_daxpy(alpha, &x, &y, &num_flops);
            if (err) {
                if (verbose > 0)
                    fprintf(diagf, "\n");
                mtxvector_free(&y);
                mtxvector_free(&x);
                return err;
            }
            if (verbose > 0) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s)\n",
                        timespec_duration(t0, t1),
                        1.0e-9 * num_flops / timespec_duration(t0, t1));
            }
        }
    } else {
        mtxvector_free(&y);
        mtxvector_free(&x);
        return MTX_ERR_INVALID_PRECISION;
    }

    if (!quiet) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        enum mtxfileformat mtxfmt =
            y.type == mtxvector_coordinate
            ? mtxfile_coordinate : mtxfile_array;
        err = mtxvector_fwrite(
            &y, mtxfmt, stdout, format, &bytes_written);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
            mtxvector_free(&x);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
        }
    }

    mtxvector_free(&y);
    mtxvector_free(&x);
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
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s ‘%s’\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        return EXIT_FAILURE;
    }

    /* 2. Read the ‘x’ vector from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfilex;
    int lines_read;
    int64_t bytes_read;
    err = mtxfile_read(
        &mtxfilex, args.precision,
        args.x_path ? args.x_path : "", args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d: %s\n",
                program_invocation_short_name,
                args.x_path, lines_read+1,
                mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.x_path, mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    /* 3. Read the ‘y’ vector from a Matrix Market file, or use a
     * vector of zeros. */
    struct mtxfile mtxfiley;
    if (args.y_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxfile_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read;
        int64_t bytes_read;
        err = mtxfile_read(
            &mtxfiley, args.precision,
            args.y_path, args.gzip,
            &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (lines_read >= 0) {
                fprintf(stderr, "%s: %s:%d: %s\n",
                        program_invocation_short_name,
                        args.y_path, lines_read+1,
                        mtxstrerror(err));
            } else {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.y_path,
                        mtxstrerror(err));
            }
            mtxfile_free(&mtxfilex);
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
        err = mtxfile_alloc_copy(&mtxfiley, &mtxfilex);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&mtxfilex);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfile_set_constant_real_single(&mtxfiley, 0.0f);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&mtxfiley);
            mtxfile_free(&mtxfilex);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    /* 3. Add the vectors. */
    if (mtxfilex.header.object == mtxfile_matrix) {
        /* TODO: Add matrices. */
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                strerror(ENOTSUP));
        mtxfile_free(&mtxfiley);
        mtxfile_free(&mtxfilex);
        program_options_free(&args);
        return EXIT_FAILURE;

    } else if (mtxfilex.header.object == mtxfile_vector) {
        err = vector_axpy(
            args.alpha, &mtxfilex, &mtxfiley, args.vector_type, args.format,
            args.repeat, args.verbose, diagf, args.quiet);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&mtxfiley);
            mtxfile_free(&mtxfilex);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

    } else {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(MTX_ERR_INVALID_MTX_OBJECT));
        mtxfile_free(&mtxfiley);
        mtxfile_free(&mtxfilex);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 4. Clean up. */
    mtxfile_free(&mtxfiley);
    mtxfile_free(&mtxfilex);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
#endif
