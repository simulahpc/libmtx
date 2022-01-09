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
 * Last modified: 2022-01-07
 *
 * Compute the Euclidean norm of a vector.
 *
 * ‘nrm2 := x'*x’,
 *
 * where ‘x'’ denotes the transpose of the vector ‘x’.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#ifdef LIBMTX_HAVE_MPI
#include <mpi.h>
#endif

#include <errno.h>

#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxnrm2";
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
    char * x_path;
    char * format;
    enum mtxprecision precision;
    enum mtxvectortype vector_type;
    bool gzip;
    int num_parts;
    enum mtxpartitioning partition;
    char * partition_path;
    int verbose;
    bool quiet;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->x_path = NULL;
    args->format = NULL;
    args->precision = mtx_double;
    args->vector_type = mtxvector_auto;
    args->gzip = false;
    args->num_parts = 1;
    args->partition = mtx_block;
    args->partition_path = NULL;
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
    if(args->format)
        free(args->format);
    if (args->partition_path)
        free(args->partition_path);
}

/**
 * `program_options_print_usage()` prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] x\n", program_name);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Compute the Euclidean norm of a vector.\n");
    fprintf(f, "\n");
    fprintf(f, " The operation performed is ‘nrm2 := x'*x’, where\n");
    fprintf(f, " ‘x’ is a vector and ‘x'’ denotes its transpose.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  x\tPath to Matrix Market file for the vector x.\n");
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
    fprintf(f, "  --parts=N\t\tnumber of parts to use for partitioning\n");
    fprintf(f, "  --partition=TYPE\tmethod of partitioning vector: ‘block’, ‘cyclic’,\n");
    fprintf(f, "\t\t\t‘block-cyclic’, ‘singleton’ or ‘unstructured’.\n");
    fprintf(f, "\t\t\t(default: ‘block’)\n");
    fprintf(f, "  --partition-path=FILE\tpath to Matrix Market file for reading partition\n");
    fprintf(f, "\t\t\twhen the partition is ‘unstructured’.\n");
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

        /* Parse partitioning options. */
        if (strcmp((*argv)[0], "--parts") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->num_parts, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--parts=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--parts="), NULL,
                &args->num_parts, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--partition") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtx_parse_partition_type(
                &args->partition, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--partition=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--partition=");
            err = mtx_parse_partition_type(
                &args->partition, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--partition-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->partition_path)
                free(args->partition_path);
            args->partition_path = strdup((*argv)[1]);
            if (!args->partition_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--partition-path=") == (*argv)[0]) {
            if (args->partition_path)
                free(args->partition_path);
            args->partition_path =
                strdup((*argv)[0] + strlen("--partition-path="));
            if (!args->partition_path) {
                program_options_free(args);
                return errno;
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
        if (strcmp((*argv)[0], "--version") == 0) {
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
        if (strcmp((*argv)[0], "--") == 0) {
            (*argc)--; (*argv)++;
            break;
        }

        /* Unrecognised option. */
        if (strlen((*argv)[0]) > 1 && (*argv)[0][0] == '-') {
            program_options_free(args);
            return EINVAL;
        }

        /*
         * Parse positional arguments.
         */
        if (num_positional_arguments_consumed == 0) {
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
 * `distvector_nrm2()' converts a Matrix Market file to a vector of the
 * given type, computes the Euclidean norm and prints the result to
 * standard output.
 */
static int distvector_nrm2(
    struct mtxdistfile * mtxdistfile,
    const struct mtxpartition * partition,
    enum mtxvectortype vector_type,
    const char * format,
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
    enum mtxprecision precision = mtxdistfile->precision;

    /* 1. Convert Matrix Market file to a vector. */
    if (verbose > 0) {
        fprintf(diagf, "mtxdistvector_from_mtxdistfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxdistvector x;
    err = mtxdistvector_from_mtxdistfile(
        &x, mtxdistfile, vector_type, comm, disterr);
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

    /* 2. Compute the Euclidean norm. */
    if (precision == mtx_single) {
        if (verbose > 0) {
            fprintf(diagf, "mtxdistvector_snrm2: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        float nrm2 = 0.0f;
        int64_t num_flops = 0;
        err = mtxdistvector_snrm2(&x, &nrm2, &num_flops, disterr);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
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
        if (!quiet) {
            fprintf(stdout, format ? format : "%.*g", FLT_DIG, nrm2);
            fputc('\n', stdout);
        }
    } else if (precision == mtx_double) {
        if (verbose > 0) {
            fprintf(diagf, "mtxdistvector_dnrm2: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        double nrm2 = 0.0;
        int64_t num_flops = 0;
        err = mtxdistvector_dnrm2(&x, &nrm2, &num_flops, disterr);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
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
        if (!quiet) {
            fprintf(stdout, format ? format : "%.*g", DBL_DIG, nrm2);
            fputc('\n', stdout);
        }
    } else {
        mtxdistvector_free(&x);
        return MTX_ERR_INVALID_PRECISION;
    }
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
    int argc_copy = argc;
    char ** argv_copy = argv;
    err = parse_program_options(&argc_copy, &argv_copy, &args);
    if (err) {
        if (rank == root) {
            fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                    strerror(err), argv_copy[0]);
        }
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank != root) {
        args.verbose = false;
        args.quiet = true;
    }
    if (args.partition == mtx_unstructured && !args.partition_path) {
        if (rank == root) {
            fprintf(stderr, "%s: Please specify a Matrix Market file "
                    "with --partition-path\n",
                    program_invocation_short_name);
        }
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (comm_size < args.num_parts) {
        if (rank == root) {
            fprintf(stderr, "%s: The number of MPI processes (%d) must "
                    "equal or exceed the number of parts in the partition (%d).\n",
                    program_invocation_short_name,
                    comm_size, args.num_parts);
        }
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 3. Read the `x' vector from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxdistfile_read_shared: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxdistfile mtxdistfile;
    int lines_read;
    int64_t bytes_read;
    err = mtxdistfile_read_shared(
        &mtxdistfile, args.precision,
        args.x_path ? args.x_path : "", args.gzip,
        &lines_read, &bytes_read,
        comm, &disterr);
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

    /* 3. Partition the vector. */
    struct mtxpartition partition;
    if (args.partition == mtx_unstructured) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_read_parts: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxpartition_read_parts(
            &partition, args.num_parts, args.partition_path,
            &lines_read, &bytes_read);
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root && lines_read >= 0) {
                fprintf(stderr, "%s: %s:%d: %s\n",
                        program_invocation_short_name,
                        args.partition_path, lines_read+1,
                        mtxdisterror_description(&disterr));
            } else if (rank == root) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.partition_path, mtxdisterror_description(&disterr));
            }
            mtxdistfile_free(&mtxdistfile);
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
            fflush(diagf);
        }

    } else {
        if (args.verbose > 0) {
            fprintf(diagf, "mtxpartition_init: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtxpartition_init(
            &partition, args.partition,
            mtxdistfile.size.num_rows, args.num_parts, 0, NULL);
        if (mtxdisterror_allreduce(&disterr, err)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        mtxdisterror_description(&disterr));
            }
            mtxdistfile_free(&mtxdistfile);
            program_options_free(&args);
            mtxdisterror_free(&disterr);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 4. Compute the Euclidean norm. */
    if (mtxdistfile.header.object == mtxfile_matrix) {
        /* TODO: Compute the Frobenius norm of the matrix. */
        if (rank == root) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    strerror(ENOTSUP));
        }
        mtxdistfile_free(&mtxdistfile);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;

    } else if (mtxdistfile.header.object == mtxfile_vector) {
        err = distvector_nrm2(
            &mtxdistfile, &partition, args.vector_type,
            args.format, args.verbose, diagf, args.quiet,
            comm, comm_size, rank, root, &disterr);
        if (err) {
            if (rank == root) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        err == MTX_ERR_MPI_COLLECTIVE
                        ? mtxdisterror_description(&disterr)
                        : mtxstrerror(err));
            }
            mtxdistfile_free(&mtxdistfile);
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
        mtxdistfile_free(&mtxdistfile);
        program_options_free(&args);
        mtxdisterror_free(&disterr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 5. Clean up. */
    mtxdistfile_free(&mtxdistfile);
    program_options_free(&args);
    mtxdisterror_free(&disterr);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
#else
/**
 * `vector_nrm2()' converts a Matrix Market file to a vector of the
 * given type, computes the Euclidean norm and prints the result to
 * standard output.
 */
static int vector_nrm2(
    struct mtxfile * mtxfile,
    enum mtxvectortype vector_type,
    const char * format,
    int verbose,
    FILE * diagf,
    bool quiet)
{
    int err;
    struct timespec t0, t1;
    enum mtxprecision precision = mtxfile->precision;

    /* 1. Convert Matrix Market file to a vector. */
    if (verbose > 0) {
        fprintf(diagf, "mtxvector_from_mtxfile: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxvector x;
    err = mtxvector_from_mtxfile(&x, mtxfile, vector_type);
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

    /* 2. Compute the Euclidean norm. */
    if (precision == mtx_single) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_snrm2: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        float nrm2 = 0.0f;
        int64_t num_flops = 0;
        err = mtxvector_snrm2(&x, &nrm2, &num_flops);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
            mtxvector_free(&x);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-9 * num_flops / timespec_duration(t0, t1));
        }
        if (!quiet) {
            fprintf(stdout, format ? format : "%.*g", FLT_DIG, nrm2);
            fputc('\n', stdout);
        }
    } else if (precision == mtx_double) {
        if (verbose > 0) {
            fprintf(diagf, "mtxvector_dnrm2: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        double nrm2 = 0.0;
        int64_t num_flops = 0;
        err = mtxvector_dnrm2(&x, &nrm2, &num_flops);
        if (err) {
            if (verbose > 0)
                fprintf(diagf, "\n");
            mtxvector_free(&x);
            return err;
        }
        if (verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds (%'.3f Gflop/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-9 * num_flops / timespec_duration(t0, t1));
        }
        if (!quiet) {
            fprintf(stdout, format ? format : "%.*g", DBL_DIG, nrm2);
            fputc('\n', stdout);
        }
    } else {
        mtxvector_free(&x);
        return MTX_ERR_INVALID_PRECISION;
    }

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
    int argc_copy = argc;
    char ** argv_copy = argv;
    err = parse_program_options(&argc_copy, &argv_copy, &args);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv_copy[0]);
        return EXIT_FAILURE;
    }

    /* 2. Read the `x' vector from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    int lines_read;
    int64_t bytes_read;
    err = mtxfile_read(
        &mtxfile, args.precision,
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

    /* 3. Compute the Euclidean norm. */
    if (mtxfile.header.object == mtxfile_matrix) {
        /* TODO: Compute the Frobenius norm of the matrix. */
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                strerror(ENOTSUP));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;

    } else if (mtxfile.header.object == mtxfile_vector) {
        err = vector_nrm2(
            &mtxfile, args.vector_type, args.format,
            args.verbose, diagf, args.quiet);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtxstrerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

    } else {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtxstrerror(MTX_ERR_INVALID_MTX_OBJECT));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 4. Clean up. */
    mtxfile_free(&mtxfile);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
#endif
