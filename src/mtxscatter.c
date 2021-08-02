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
 * Scatter a Matrix Market object to one or more MPI processes.
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

const char * program_name = "mtxscatter";
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
    char * mtx_path;
    bool gzip;
    char * mtx_output_path;
    char * format;
    int verbose;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->mtx_path = NULL;
    args->gzip = false;
    args->mtx_output_path = strdup("out%p.mtx");
    args->format = NULL;
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
    if (args->mtx_path)
        free(args->mtx_path);
    if (args->mtx_output_path)
        free(args->mtx_output_path);
    if (args->format)
        free(args->format);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Scatter a Matrix Market object among MPI processes\n");
    fprintf(f, " and write the results to file.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --output-path=FILE\tpath for scattered Matrix Market files, where\n");
    fprintf(f, "\t\t\t  '%%p' in the string is replaced with the rank of\n");
    fprintf(f, "\t\t\t  of each MPI process (default: out%%p.mtx).\n");
    fprintf(f, "  --format=FORMAT\tFormat string for outputting numerical values.\n");
    fprintf(f, "\t\t\tFor real, double and complex values, the format specifiers\n");
    fprintf(f, "\t\t\t'%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "\t\t\twhereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "\t\t\tand precision can optionally be specified, e.g., \"%%+3.1f\".\n");
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

        /* Parse output path. */
        if (strcmp((*argv)[0], "--output-path") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            if (args->mtx_output_path)
                free(args->mtx_output_path);
            args->mtx_output_path = strdup((*argv)[1]);
            if (!args->mtx_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--output-path=") == (*argv)[0]) {
            if (args->mtx_output_path)
                free(args->mtx_output_path);
            args->mtx_output_path =
                strdup((*argv)[0] + strlen("--output-path="));
            if (!args->mtx_output_path) {
                program_options_free(args);
                return errno;
            }
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
 * `num_places()` is the number of digits or places in a given
 * non-negative integer.
 */
static int num_places(int n, int * places)
{
    int r = 1;
    if (n < 0)
        return EINVAL;
    while (n > 9) {
        n /= 10;
        r++;
    }
    *places = r;
    return 0;
}

/**
 * `format_path()` formats a path by replacing any occurence of '%p'
 * in `path_fmt' with the rank of the MPI process.
 */
static int format_path(
    const char * path_fmt,
    char ** output_path,
    int rank,
    int comm_size)
{
    int err;
    int rank_places;
    err = num_places(comm_size, &rank_places);
    if (err)
        return err;

    /* Count the number of occurences of '%p' in the format string. */
    int count = 0;
    const char * needle = "%p";
    const int needle_len = strlen(needle);
    const int path_fmt_len = strlen(path_fmt);

    const char * src = path_fmt;
    const char * next;
    while ((next = strstr(src, needle))) {
        count++;
        src = next + needle_len;
        assert(src < path_fmt + path_fmt_len);
    }

    /* Allocate storage for the path. */
    int path_len = path_fmt_len + (rank_places-needle_len)*count;
    char * path = malloc(path_len+1);
    if (!path)
        return errno;
    path[path_len] = '\0';

    src = path_fmt;
    char * dest = path;
    while ((next = strstr(src, needle))) {
        /* Copy the format string up until the needle, '%p'. */
        while (src < next && dest <= path + path_len)
            *dest++ = *src++;
        src += needle_len;

        /* Replace '%p' with the current MPI rank. */
        assert(dest + rank_places <= path + path_len);
        int len = snprintf(dest, rank_places+1, "%0*d", rank_places, rank);
        assert(len == rank_places);
        dest += rank_places;
    }

    /* Copy the remainder of the format string. */
    while (*src != '\0' && dest <= path + path_len)
        *dest++ = *src++;
    assert(dest == path + path_len);
    *dest = '\0';

    *output_path = path;
    return 0;
}

/**
 * `partition_rows()` partitions a matrix or vector by rows.
 *
 * The arrays `row_sets` and `column_sets` will contain the index sets
 * for the rows and columns, respectively, that are assigned to each
 * MPI process.
 */
static int partition_rows(
    int num_rows,
    int num_columns,
    MPI_Comm comm,
    int comm_size,
    int rank,
    struct mtx_index_set ** row_sets,
    struct mtx_index_set ** column_sets)
{
    int err;

    /* Allocate storage for the index sets. */
    *row_sets = (struct mtx_index_set *) malloc(
        comm_size * sizeof(struct mtx_index_set));
    if (!*row_sets)
        return MTX_ERR_ERRNO;
    *column_sets = (struct mtx_index_set *) malloc(
        comm_size * sizeof(struct mtx_index_set));
    if (!*column_sets) {
        free(*row_sets);
        return MTX_ERR_ERRNO;
    }

    /* Partition the rows into equal-sized, contiguous parts. */
    int num_rows_per_process = (num_rows + comm_size - 1) / comm_size;
    for (int p = 0; p < comm_size; p++) {
        int local_row_start = 1 + p * num_rows_per_process;
        int local_row_end = 1 + (p + 1) * num_rows_per_process;
        if (local_row_end > num_rows+1)
            local_row_end = num_rows+1;
        err = mtx_index_set_init_interval(
            &(*row_sets)[p], local_row_start, local_row_end);
        if (err) {
            for (int q = p-1; q >= 0; q--) {
                mtx_index_set_free(&(*column_sets)[p]);
                mtx_index_set_free(&(*row_sets)[p]);
            }
            free(*column_sets);
            free(*row_sets);
            return err;
        }

        /* Every process is given ownership of every column. */
        int local_column_start = 1;
        int local_column_end = num_columns+1;
        err = mtx_index_set_init_interval(
            &(*column_sets)[p], local_column_start, local_column_end);
        if (err) {
            for (int q = p-1; q >= 0; q--) {
                mtx_index_set_free(&(*column_sets)[p]);
                mtx_index_set_free(&(*row_sets)[p]);
            }
            free(*column_sets);
            free(*row_sets);
            return err;
        }
    }
    return 0;
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
    if (!args.mtx_path) {
        fprintf(stderr, "%s: Please specify a Matrix Market file\n",
                program_invocation_short_name);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 2. Initialise MPI. */
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int root = 0;
    mpierr = MPI_Init(&argc, &argv);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    mpierr = MPI_Barrier(comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Get the MPI rank of the current process. */
    int rank;
    mpierr = MPI_Comm_rank(comm, &rank);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    mpierr = MPI_Barrier(comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Get the size of the MPI communicator. */
    int comm_size;
    mpierr = MPI_Comm_size(comm, &comm_size);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    mpierr = MPI_Barrier(comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* 1. Read a Matrix Market file on the MPI root process. */
    struct mtx mtx;
    if (rank == root) {
        if (args.verbose > 0) {
            fprintf(diagf, "mtx_read: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int line_number, column_number;
        err = mtx_read(
            &mtx, args.mtx_path, args.gzip,
            &line_number, &column_number);
        if (err && (line_number == -1 && column_number == -1)) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.mtx_path, mtx_strerror(err));
            program_options_free(&args);
            MPI_Abort(comm, EXIT_FAILURE);
        } else if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s:%d:%d: %s\n",
                    program_invocation_short_name,
                    args.mtx_path, line_number, column_number,
                    mtx_strerror(err));
            program_options_free(&args);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds "
                    "%s object %s format %s field %s symmetry "
                    "%d rows %d columns %"PRId64" nonzeros\n",
                    timespec_duration(t0, t1),
                    mtx_object_str(mtx.object),
                    mtx_format_str(mtx.format),
                    mtx_field_str(mtx.field),
                    mtx_symmetry_str(mtx.symmetry),
                    mtx.num_rows,
                    mtx.num_columns,
                    mtx.num_nonzeros);
        }
    }
    mpierr = MPI_Barrier(comm);
    if (mpierr) {
        MPI_Error_string(mpierr, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Barrier failed with %s\n",
                program_invocation_short_name, mpierrstr);
        if (rank == root)
            mtx_free(&mtx);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* 2. Scatter the matrix among MPI ranks. */
    if (rank == root && args.verbose > 0) {
        fprintf(diagf, "mtx_matrix_coordinate_scatter: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* Partition rows and columns among MPI processes. */
    struct mtx_index_set * row_sets;
    struct mtx_index_set * column_sets;
    err = partition_rows(
        mtx.num_rows, mtx.num_columns,
        comm, comm_size, rank, &row_sets, &column_sets);
    if (err) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                strerror(err));
        if (rank == root)
            mtx_free(&mtx);
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
        for (int q = 0; q < comm_size; q++) {
            mtx_index_set_free(&row_sets[q]);
            mtx_index_set_free(&column_sets[q]);
        }
        free(row_sets);
        free(column_sets);
        if (rank == root)
            mtx_free(&mtx);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* Scatter the matrix. */
    struct mtx dstmtx;
    err = mtx_matrix_coordinate_scatter(
        &dstmtx, &mtx, row_sets, column_sets, comm, root, &mpierr);
    if (err) {
        if (rank == root && args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s\n",
                program_invocation_short_name,
                mtx_strerror_mpi(err, mpierr, mpierrstr));
        for (int q = 0; q < comm_size; q++) {
            mtx_index_set_free(&row_sets[q]);
            mtx_index_set_free(&column_sets[q]);
        }
        free(row_sets);
        free(column_sets);
        if (rank == root)
            mtx_free(&mtx);
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
        for (int q = 0; q < comm_size; q++) {
            mtx_index_set_free(&row_sets[q]);
            mtx_index_set_free(&column_sets[q]);
        }
        free(row_sets);
        free(column_sets);
        if (rank == root)
            mtx_free(&mtx);
        program_options_free(&args);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == root && args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%.6f seconds\n", timespec_duration(t0, t1));
    }

    /* Free the original Matrix Market object. */
    for (int q = 0; q < comm_size; q++) {
        mtx_index_set_free(&row_sets[q]);
        mtx_index_set_free(&column_sets[q]);
    }
    free(row_sets);
    free(column_sets);
    if (rank == root)
        mtx_free(&mtx);

    /* 3. Write the Matrix Market object of each MPI process to file. */
    if (args.mtx_output_path) {
        if (rank == root && args.verbose > 0) {
            fprintf(diagf, "mtx_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* Format the output path. */
        char * output_path;
        err = format_path(
            args.mtx_output_path, &output_path,
            rank, comm_size);
        if (err) {
            if (rank == root && args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, strerror(err),
                    output_path);
            mtx_free(&dstmtx);
            program_options_free(&args);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        /* Write scattered Matrix Market objects to file. */
        err = mtx_write(&dstmtx, output_path, args.gzip, args.format);
        if (err) {
            if (rank == root && args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    output_path, mtx_strerror(err));
            free(output_path);
            mtx_free(&dstmtx);
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
            free(output_path);
            mtx_free(&dstmtx);
            program_options_free(&args);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        free(output_path);
        if (rank == root && args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    }

    /* 4. Clean up. */
    mtx_free(&dstmtx);
    program_options_free(&args);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
