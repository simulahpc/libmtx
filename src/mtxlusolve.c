/* This file is part of libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-03
 *
 * Solve a linear system of equation `Ax=b' using a LU
 * factorisation-based direct solver from SuperLU_DIST.
 */

#include <matrixmarket/matrixmarket.h>

#include "../matrixmarket/parse.h"

#include <mpi.h>

#include <errno.h>

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxlusolve";
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
    char * A_path;
    char * b_path;
    char * format;
    bool gzip;
    int verbose;
    bool quiet;

    /* SuperLU_DIST options. */
    int num_process_rows;
    int num_process_columns;
    enum mtx_superlu_dist_fact fact;
    bool equil;
    bool parsymbfact;
    enum mtx_superlu_dist_colperm colperm;
    enum mtx_superlu_dist_rowperm rowperm;
    bool replacetinypivot;
    enum mtx_superlu_dist_iterrefine iterrefine;
    enum mtx_superlu_dist_trans trans;
    bool solveinitialized;
    bool refineinitialized;
    bool printstat;
    int num_lookaheads;
    bool lookahead_etree;
    bool sympattern;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->A_path = NULL;
    args->b_path = NULL;
    args->format = NULL;
    args->gzip = false;
    args->quiet = false;
    args->verbose = 0;

    /* SuperLU_DIST options. */
    args->num_process_rows = 1;
    args->num_process_columns = 1;
    args->fact = mtx_superlu_dist_fact_DOFACT;
    args->equil = true;
    args->parsymbfact = false;
    args->colperm = mtx_superlu_dist_colperm_METIS_AT_PLUS_A;
    args->rowperm = mtx_superlu_dist_rowperm_LargeDiag_MC64;
    args->replacetinypivot = false;
    args->iterrefine = mtx_superlu_dist_iterrefine_DOUBLE;
    args->trans = mtx_superlu_dist_trans_NOTRANS;
    args->solveinitialized = false;
    args->refineinitialized = false;
    args->printstat = true;
    args->num_lookaheads = 10;
    args->lookahead_etree = false;
    args->sympattern = false;
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
    if(args->format)
        free(args->format);
}

/**
 * `program_options_print_usage()` prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A [b]\n", program_name);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Solve a linear system of equation `Ax=b' using a LU\n");
    fprintf(f, " factorisation-based direct solver from SuperLU_DIST.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  A\tPath to Matrix Market file for the matrix A.\n");
    fprintf(f, "  b\tOptional path to Matrix Market file for the vector b.\n");
    fprintf(f, "   \tIf omitted, then a vector of ones is used.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
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
    fprintf(f, " Options related to SuperLU_DIST are:\n");
    fprintf(f, "  --process-rows=N\tnumber of rows in 2D process grid\n");
    fprintf(f, "  --process-columns=N\tnumber of columns in 2D process grid\n");
    fprintf(f, "  --no-equil\t\tdisable equilibration\n");
    fprintf(f, "  --equil\t\tenable equilibration (default)\n");
    fprintf(f, "  --no-parsymbfact\tdo not perform symbolic factorisation in parallel (default)\n");
    fprintf(f, "  --parsymbfact\t\tperform symbolic factorisation in parallel\n");
    fprintf(f, "  --colperm=PERM\tcolumn permutation, choose one of: NATURAL,\n");
    fprintf(f, "\t\t\tMETIS_AT_PLUS_A, PARMETIS, MMD_ATA, MMD_AT_PLUS_A, COLAMD\n");
    fprintf(f, "\t\t\tand MY_PERMC. The default is METIS_AT_PLUS_A if ParMETIS\n");
    fprintf(f, "\t\t\tis available, or MMD_AT_PLUS_A otherwise.\n");
    fprintf(f, "  --rowperm=PERM\trow permutation, choose one of: NO, LargeDiag_MC64\n");
    fprintf(f, "\t\t\tLargeDiag_AWPM and MY_PERMR. The default is LargeDiag_MC64\n");
    fprintf(f, "  --no-replacetinypivot\tdo not replace tiny diagonals during factorisation (default)\n");
    fprintf(f, "  --replacetinypivot\treplace tiny diagonals during factorisation\n");
    fprintf(f, "  --iterrefine=METHOD\tspecify how to perform iterative refinement. Choose one of:\n");
    fprintf(f, "\t\t\tNO, SINGLE and DOUBLE. The default is DOUBLE.\n");
    fprintf(f, "  --trans=TRANS\t\twhether to solve the transposed system. Choose one of:\n");
    fprintf(f, "\t\t\tNOTRANS, TRANS or CONJ. The default is NOTRANS.\n");
    fprintf(f, "  --no-printstat\tdo not print solver statistics\n");
    fprintf(f, "  --printstat\t\tprint solver statistics (default)\n");
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
                errno = EINVAL;
                return MTX_ERR_ERRNO;
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
         * Parse SuperLU_DIST options.
         */
        if (strcmp((*argv)[0], "--process-rows") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                errno = EINVAL;
                return MTX_ERR_ERRNO;
            }
            err = parse_int32((*argv)[1], NULL, &args->num_process_rows, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--process-rows=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--process-rows="), NULL,
                &args->num_process_rows, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--process-columns") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                errno = EINVAL;
                return MTX_ERR_ERRNO;
            }
            err = parse_int32((*argv)[1], NULL, &args->num_process_columns, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--process-columns=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--process-columns="), NULL,
                &args->num_process_columns, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--equil") == 0) {
            args->equil = true;
            num_arguments_consumed++;
            continue;
        } else if (strcmp((*argv)[0], "--no-equil") == 0) {
            args->equil = false;
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--parsymbfact") == 0) {
            args->parsymbfact = true;
            num_arguments_consumed++;
            continue;
        } else if (strcmp((*argv)[0], "--no-parsymbfact") == 0) {
            args->parsymbfact = false;
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--colperm") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                errno = EINVAL;
                return MTX_ERR_ERRNO;
            }
            err = mtx_superlu_dist_colperm_parse(
                (*argv)[1], &args->colperm, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--colperm=") == (*argv)[0]) {
            err = mtx_superlu_dist_colperm_parse(
                (*argv)[0] + strlen("--colperm="),
                &args->colperm, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--rowperm") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                errno = EINVAL;
                return MTX_ERR_ERRNO;
            }
            err = mtx_superlu_dist_rowperm_parse(
                (*argv)[1], &args->rowperm, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--rowperm=") == (*argv)[0]) {
            err = mtx_superlu_dist_rowperm_parse(
                (*argv)[0] + strlen("--rowperm="),
                &args->rowperm, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--replacetinypivot") == 0) {
            args->replacetinypivot = true;
            num_arguments_consumed++;
            continue;
        } else if (strcmp((*argv)[0], "--no-replacetinypivot") == 0) {
            args->replacetinypivot = false;
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--iterrefine") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                errno = EINVAL;
                return MTX_ERR_ERRNO;
            }
            err = mtx_superlu_dist_iterrefine_parse(
                (*argv)[1], &args->iterrefine, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--iterrefine=") == (*argv)[0]) {
            err = mtx_superlu_dist_iterrefine_parse(
                (*argv)[0] + strlen("--iterrefine="),
                &args->iterrefine, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--trans") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                errno = EINVAL;
                return MTX_ERR_ERRNO;
            }
            err = mtx_superlu_dist_trans_parse(
                (*argv)[1], &args->trans, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--trans=") == (*argv)[0]) {
            err = mtx_superlu_dist_trans_parse(
                (*argv)[0] + strlen("--trans="),
                &args->trans, NULL, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--printstat") == 0) {
            args->printstat = true;
            num_arguments_consumed++;
            continue;
        } else if (strcmp((*argv)[0], "--no-printstat") == 0) {
            args->printstat = false;
            num_arguments_consumed++;
            continue;
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
        } else {
            program_options_free(args);
            errno = EINVAL;
            return MTX_ERR_ERRNO;
        }

        num_positional_arguments_consumed++;
        num_arguments_consumed++;
    }

    if (num_positional_arguments_consumed < 1) {
        program_options_free(args);
        program_options_print_usage(stdout);
        exit(EXIT_FAILURE);
    }

    return MTX_SUCCESS;
}

/**
 * `mpi_allgather_err()' is used to gather return codes from all MPI
 * processes in the communicator `comm', write them to the array
 * `errs', and then set the value of `err' for every MPI process if
 * any of the return codes from any of the processes are nonzero.
 */
static int mpi_allgather_err(
    MPI_Comm comm,
    int comm_size,
    int err,
    int * errs)
{
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    mpierr = MPI_Allgather(&err, 1, MPI_INT, errs, 1, MPI_INT, comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Allgather failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    for (int i = 0; i < comm_size; i++) {
        if (errs[i])
            return errs[i];
    }
    return MTX_SUCCESS;
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
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    struct timespec t0, t1;
    FILE * diagf = stderr;

    /* 1. Initialise MPI. */
    const MPI_Comm world_comm = MPI_COMM_WORLD;
    const int root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        return EXIT_FAILURE;
    }

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(world_comm, &rank);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    /* Get the size of the MPI communicator. */
    int world_comm_size;
    mpierr = MPI_Comm_size(world_comm, &world_comm_size);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    /* Allocate storage for error handling. */
    int * errs = alloca(world_comm_size * sizeof(int));
    if (!errs) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(errno));
        MPI_Abort(world_comm, EXIT_FAILURE);
    }
    for (int i = 0; i < world_comm_size; i++)
        errs[i] = 0;

    /* 2. Parse program options. */
    struct program_options args;
    int argc_copy = argc;
    char ** argv_copy = argv;
    err = parse_program_options(&argc_copy, &argv_copy, &args);
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (errs[rank]) {
            fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                    mtx_strerror(err), argv_copy[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 3. Read the matrix from a Matrix Market file. */
    struct mtx A;
    int line_number, column_number;
    if (rank == root) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtx_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtx_read(
            &A, args.A_path ? args.A_path : "", args.gzip,
            &line_number, &column_number);
    }
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        if (errs[rank] && line_number == -1 && column_number == -1) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.A_path, mtx_strerror(err));
        } else if (errs[rank]) {
            fprintf(stderr, "%s: %s:%d:%d: %s\n",
                    program_invocation_short_name,
                    args.A_path, line_number, column_number,
                    mtx_strerror(err));
        }
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* If necessary, sort the matrix in column major order. */
    if (rank == root && A.sorting != mtx_column_major) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtx_sort: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtx_sort(&A, mtx_column_major);
    }
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        if (errs[rank]) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                    mtx_strerror(err));
        }
        if (rank == root)
            mtx_free(&A);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n",
                timespec_duration(t0, t1));
        fprintf(diagf, "mtx_bcast: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* Broadcast matrix to all MPI processes. */
    err = mtx_bcast(&A, root, world_comm, &mpierr);
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        if (errs[rank]) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                    mtx_strerror_mpi(err, mpierr, mpierrstr));
        }
        if (rank == root)
            mtx_free(&A);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 4. Read the vector b from a Matrix Market file, or use a vector
     * of all ones. */
    struct mtx b;
    if (args.b_path && strlen(args.b_path) > 0) {
        if (rank == root) {
            if (args.verbose > 0) {
                fprintf(diagf, "mtx_read: ");
                fflush(diagf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
            }

            err = mtx_read(
                &b, args.b_path ? args.b_path : "", args.gzip,
                &line_number, &column_number);
        }
        if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
            if (rank == root && args.verbose > 0)
                fprintf(diagf, "\n");
            if (errs[rank] && line_number == -1 && column_number == -1) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        args.b_path, mtx_strerror(err));
            } else if (errs[rank]) {
                fprintf(stderr, "%s: %s:%d:%d: %s\n",
                        program_invocation_short_name,
                        args.b_path, line_number, column_number,
                        mtx_strerror(err));
            }
            mtx_free(&A);
            program_options_free(&args);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (rank == root && args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n",
                    timespec_duration(t0, t1));
            fprintf(diagf, "mtx_bcast: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* Broadcast vector to all MPI processes. */
        err = mtx_bcast(&b, root, world_comm, &mpierr);
        if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
            if (rank == root && args.verbose > 0)
                fprintf(diagf, "\n");
            if (errs[rank]) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                        mtx_strerror_mpi(err, mpierr, mpierrstr));
            }
            if (rank == root)
                mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (rank == root && args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    } else {
        if (A.field == mtx_real) {
            err = mtx_alloc_vector_array_real(
                &b, 0, NULL, A.num_columns);
            if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
                if (errs[rank]) {
                    fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                            mtx_strerror(err));
                }
                mtx_free(&A);
                program_options_free(&args);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            err = mtx_set_constant_real(&b, 1.0f);
            if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
                if (errs[rank]) {
                    fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                            mtx_strerror(err));
                }
                mtx_free(&b);
                mtx_free(&A);
                program_options_free(&args);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        } else if (A.field == mtx_double) {
            err = mtx_alloc_vector_array_double(
                &b, 0, NULL, A.num_columns);
            if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
                if (errs[rank]) {
                    fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                            mtx_strerror(err));
                }
                mtx_free(&A);
                program_options_free(&args);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            err = mtx_set_constant_double(&b, 1.0);
            if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
                if (errs[rank]) {
                    fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                            mtx_strerror(err));
                }
                mtx_free(&b);
                mtx_free(&A);
                program_options_free(&args);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    strerror(ENOTSUP));
            mtx_free(&A);
            program_options_free(&args);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    /* 5. Create an output vector. */
    struct mtx x;
    if (A.field == mtx_real) {
        err = mtx_alloc_vector_array_real(
            &x, 0, NULL, A.num_rows);
        if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
            if (errs[rank]) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                        mtx_strerror(err));
            }
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    } else if (A.field == mtx_double) {
        err = mtx_alloc_vector_array_double(
            &x, 0, NULL, A.num_rows);
        if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
            if (errs[rank]) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                        mtx_strerror(err));
            }
            mtx_free(&b);
            mtx_free(&A);
            program_options_free(&args);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                strerror(ENOTSUP));
        mtx_free(&b);
        mtx_free(&A);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    err = mtx_set_zero(&x);
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (errs[rank]) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                    mtx_strerror(err));
        }
        mtx_free(&x);
        mtx_free(&b);
        mtx_free(&A);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* 6. Solve the linear system `Ax=b'. */
    if (rank == root && args.verbose > 0) {
        fprintf(diagf, "mtx_superlu_dist_solve_global: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    err = mtx_superlu_dist_solve_global(
        &A, &b, &x,
        args.verbose,
        diagf,
        &mpierr,
        world_comm,
        root,
        args.num_process_rows,
        args.num_process_columns,
        args.fact,
        args.equil,
        args.parsymbfact,
        args.colperm,
        args.rowperm,
        args.replacetinypivot,
        args.iterrefine,
        args.trans,
        args.solveinitialized,
        args.refineinitialized,
        args.printstat,
        args.num_lookaheads,
        args.lookahead_etree,
        args.sympattern);
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        if (errs[rank]) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                    mtx_strerror_mpi(err, mpierr, mpierrstr));
        }
        mtx_free(&x);
        mtx_free(&b);
        mtx_free(&A);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n",
                timespec_duration(t0, t1));
    }

    /* 7. Write the result to standard output. */
    if (!args.quiet && rank == root) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtx_fwrite: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = mtx_fwrite(&x, stdout, args.format);
    }
    if (mpi_allgather_err(world_comm, world_comm_size, err, errs)) {
        if (rank == root && args.verbose > 0) {
            fflush(stdout);
            fprintf(diagf, "\n");
        }
        if (errs[rank]) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                    mtx_strerror(err));
        }
        mtx_free(&x);
        mtx_free(&b);
        mtx_free(&A);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (!args.quiet && rank == root && args.verbose > 0) {
        fflush(stdout);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n", timespec_duration(t0, t1));
        fflush(diagf);
    }

    /* 8. Clean up. */
    mtx_free(&x);
    mtx_free(&b);
    mtx_free(&A);
    program_options_free(&args);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
