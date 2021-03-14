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
 * Last modified: 2021-06-18
 *
 * Gather a Matrix Market object from one or more MPI processes.
 */

#include <matrixmarket/matrixmarket.h>

#include "../matrixmarket/parse.h"
#include "ioutil.h"

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

const char * program_name = "mtxgather";
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
    int num_paths;
    char ** mtx_paths;
    bool gzip;
    int output_field_width;
    int output_precision;
    int verbose;
    bool quiet;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->num_paths = 0;
    args->mtx_paths = NULL;
    args->gzip = false;
    args->output_field_width = 0;
    args->output_precision = -1;
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
    if (args->mtx_paths) {
        for (int i = 0; i < args->num_paths; i++)
            free(args->mtx_paths[i]);
        free(args->mtx_paths);
    }
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE..\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Read a distributed Matrix Market object from multiple files,\n");
    fprintf(f, " gather onto a single MPI process and write the result to file.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --output-width=N\tfield width for outputting numerical values\n");
    fprintf(f, "  --output-prec=N\tprecision for outputting numerical values.\n");
    fprintf(f, "\t\t\t  The default precision is 6 digits after the decimal point.\n");
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

        if (strcmp((*argv)[0], "-z") == 0 ||
            strcmp((*argv)[0], "--gzip") == 0 ||
            strcmp((*argv)[0], "--gunzip") == 0 ||
            strcmp((*argv)[0], "--ungzip") == 0)
        {
            args->gzip = true;
            num_arguments_consumed++;
            continue;
        }

        /* Parse output field width. */
        if (strcmp((*argv)[0], "--output-width") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->output_field_width, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--output-width=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--output-width="), NULL,
                &args->output_field_width, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Parse output precision. */
        if (strcmp((*argv)[0], "--output-prec") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            err = parse_int32((*argv)[1], NULL, &args->output_precision, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--output-prec=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--output-prec="), NULL,
                &args->output_precision, NULL);
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

        /* Unrecognised option. */
        if (strlen((*argv)[0]) > 1 && (*argv)[0][0] == '-') {
            program_options_free(args);
            return EINVAL;
        }

        /* Parse positional arguments. */
        char ** paths = malloc((args->num_paths+1) * sizeof(char *));
        if (!paths) {
            program_options_free(args);
            return errno;
        }
        for (int i = 0; i < args->num_paths; i++)
            paths[i] = args->mtx_paths[i];
        paths[args->num_paths] = strdup((*argv)[0]);
        args->num_paths++;
        if (args->mtx_paths)
            free(args->mtx_paths);
        args->mtx_paths = paths;
        num_positional_arguments_consumed++;
        num_arguments_consumed++;
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
    int mpierr;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
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
    if (!args.mtx_paths) {
        fprintf(stderr, "%s: Please specify one or more Matrix Market files\n",
                program_invocation_short_name);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 2. Initialise MPI. */
    const MPI_Comm world_comm = MPI_COMM_WORLD;
    const int root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    mpierr = MPI_Barrier(world_comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(world_comm, &rank);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }
    mpierr = MPI_Barrier(world_comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    /* Get the size of the MPI communicator. */
    int world_comm_size;
    mpierr = MPI_Comm_size(world_comm, &world_comm_size);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }
    mpierr = MPI_Barrier(world_comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    if (world_comm_size < args.num_paths) {
        fprintf(stderr, "%s: Please specify one Matrix Market file per MPI process.\n",
                program_invocation_short_name);
        MPI_Finalize();
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /*
     * Create a communicator whose size matches the number of input
     * files.
     */
    MPI_Group world_group;
    mpierr = MPI_Comm_group(world_comm, &world_group);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_group failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    int comm_size = args.num_paths;
    int * ranks = malloc(args.num_paths * sizeof(int));
    if (!ranks) {
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                strerror(errno));
        MPI_Finalize();
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < args.num_paths; i++)
        ranks[i] = i;

    MPI_Group group;
    mpierr = MPI_Group_incl(world_group, comm_size, ranks, &group);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Group_incl failed with %s\n",
                program_invocation_short_name, mpierrstr);
        free(ranks);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }

    MPI_Comm comm;
    mpierr = MPI_Comm_create(world_comm, group, &comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_create failed with %s\n",
                program_invocation_short_name, mpierrstr);
        free(ranks);
        program_options_free(&args);
        MPI_Abort(world_comm, EXIT_FAILURE);
    }
    free(ranks);

    if (rank >= comm_size) {
        MPI_Finalize();
        program_options_free(&args);
        return EXIT_SUCCESS;
    }

    /* 1. Read Matrix Market files on each MPI process. */
    struct mtx srcmtx;
    if (rank == root && args.verbose > 0) {
        fprintf(diagf, !args.gzip ? "mtx_read: " : "mtx_gzread: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int line_number, column_number;
    err = read_mtx(
        args.mtx_paths[rank] ? args.mtx_paths[rank] : "",
        args.gzip, &srcmtx, args.verbose,
        &line_number, &column_number);
    if (err && (line_number == -1 && column_number == -1)) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.mtx_paths[rank], mtx_strerror(err));
        MPI_Comm_free(&comm);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d:%d: %s\n",
                program_invocation_short_name,
                args.mtx_paths[rank], line_number, column_number,
                mtx_strerror(err));
        MPI_Comm_free(&comm);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds "
                "%s object %s format %s field %s symmetry "
                "%d rows %d columns %"PRId64" nonzeros\n",
                timespec_duration(t0, t1),
                mtx_object_str(srcmtx.object),
                mtx_format_str(srcmtx.format),
                mtx_field_str(srcmtx.field),
                mtx_symmetry_str(srcmtx.symmetry),
                srcmtx.num_rows,
                srcmtx.num_columns,
                srcmtx.num_nonzeros);
    }

    mpierr = MPI_Barrier(comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        mtx_free(&srcmtx);
        MPI_Comm_free(&comm);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* 2. Gather the matrix among MPI ranks. */
    if (rank == root && args.verbose > 0) {
        fprintf(diagf, "mtx_matrix_coordinate_gather: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* Gather the matrix. */
    struct mtx dstmtx;
    enum mtx_partitioning partitioning = mtx_partition;
    err = mtx_matrix_coordinate_gather(
        &dstmtx, &srcmtx, partitioning, comm, root, &mpierr);
    if (err) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtx_strerror_mpi(err, mpierr, mpierrstr));
        mtx_free(&srcmtx);
        MPI_Comm_free(&comm);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    mpierr = MPI_Barrier(comm);
    if (mpierr) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        if (rank == root)
            mtx_free(&dstmtx);
        mtx_free(&srcmtx);
        MPI_Comm_free(&comm);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n", timespec_duration(t0, t1));
    }
    mtx_free(&srcmtx);

    /* 3. Write the gathered Matrix Market object to file. */
    if (!args.quiet && rank == root) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtx_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* Write gathered Matrix Market object to file. */
        err = mtx_write(
            &dstmtx, stdout,
            args.output_field_width,
            args.output_precision);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name,
                    mtx_strerror(err));
            mtx_free(&dstmtx);
            MPI_Comm_free(&comm);
            program_options_free(&args);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 4. Clean up. */
    if (rank == root)
        mtx_free(&dstmtx);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    program_options_free(&args);
    return EXIT_SUCCESS;
}
