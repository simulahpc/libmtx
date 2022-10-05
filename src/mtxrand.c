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
 * Last modified: 2022-10-05
 *
 * Generate random matrices and vectors in Matrix Market format.
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

const char * program_name = "mtxrand";
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
    enum mtxfileobject object;
    enum mtxfileformat format;
    enum mtxfilefield field;
    enum mtxfilesymmetry symmetry;
    int64_t num_rows;
    int64_t num_columns;
    int64_t num_nonzeros;
    int64_t min;
    int64_t max;
    bool seed_given;
    int seed;
    char * numfmt;
    enum mtxprecision precision;
    int verbose;
    bool quiet;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->object = mtxfile_matrix;
    args->format = mtxfile_coordinate;
    args->field = mtxfile_real;
    args->symmetry = mtxfile_general;
    args->num_rows = 0;
    args->num_columns = 0;
    args->num_nonzeros = 0;
    args->min = 0;
    args->max = RAND_MAX;
    args->seed_given = false;
    args->seed = -1;
    args->numfmt = NULL;
    args->precision = mtx_double;
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
    if (args->numfmt) free(args->numfmt);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] M [N] [K]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Generate random matrices and vectors in Matrix Market format.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  M\tNumber of matrix rows or vector elements.\n");
    fprintf(f, "  N\tNumber of matrix columns, or number of vector nonzeros.\n");
    fprintf(f, "   \tIf omitted, this is set equal to M.\n");
    fprintf(f, "  K\tNumber of matrix nonzeros. If omitted, this is set equal to ‘M times N’.\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --object=OBJECT\tobject to generate: matrix or vector. [matrix]\n");
    fprintf(f, "  --format=FORMAT\toutput format: array or coordinate. [coordinate]\n");
    fprintf(f, "  --field=FIELD\t\tfield for numerical values: real, integer, complex or pattern. [real]\n");
    fprintf(f, "  --symmetry=SYMMETRY\tmatrix symmetry: general, symmetric, skew-symmetric or hermitian. [general]\n");
    fprintf(f, "  --precision=PRECISION\tprecision for numerical values: single or double. [double]\n");
    fprintf(f, "  --min=N\t\tlower range for randomly generated numerical values [0]\n");
    fprintf(f, "  --max=N\t\tupper range for randomly generated numerical values [%d]\n", RAND_MAX);
    fprintf(f, "  --seed=N\t\tseed for pseudo-random number generator.\n");
    fprintf(f, "\t\t\tBy default, the generator is seeded with the current time.\n");
    fprintf(f, "\n");
    fprintf(f, "  --num-fmt=FMT\t\tFormat string for outputting numerical values.\n");
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
        if (strcmp(argv[0], "--object") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxfileobject_parse(&args->object, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--object=") == argv[0]) {
            char * s = argv[0] + strlen("--object=");
            err = mtxfileobject_parse(&args->object, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--format") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxfileformat_parse(&args->format, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--format=") == argv[0]) {
            char * s = argv[0] + strlen("--format=");
            err = mtxfileformat_parse(&args->format, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--field") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxfilefield_parse(&args->field, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--field=") == argv[0]) {
            char * s = argv[0] + strlen("--field=");
            err = mtxfilefield_parse(&args->field, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--symmetry") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = mtxfilesymmetry_parse(&args->symmetry, NULL, NULL, argv[0], "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--symmetry=") == argv[0]) {
            char * s = argv[0] + strlen("--symmetry=");
            err = mtxfilesymmetry_parse(&args->symmetry, NULL, NULL, s, "");
            if (err) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

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

        if (strcmp(argv[0], "--min") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int64_ex(argv[0], NULL, &args->min, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--min=") == argv[0]) {
            err = parse_int64_ex(
                argv[0] + strlen("--min="), NULL, &args->min, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--max") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int64_ex(argv[0], NULL, &args->max, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--max=") == argv[0]) {
            err = parse_int64_ex(
                argv[0] + strlen("--max="), NULL, &args->max, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--seed") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int32_ex(argv[0], NULL, &args->seed, NULL);
            if (err) { program_options_free(args); return err; }
            args->seed_given = true;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--seed=") == argv[0]) {
            err = parse_int32_ex(
                argv[0] + strlen("--seed="), NULL, &args->seed, NULL);
            if (err) { program_options_free(args); return err; }
            args->seed_given = true;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--num-fmt") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            args->numfmt = strdup(argv[0]);
            if (!args->numfmt) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--num-fmt=") == argv[0]) {
            args->numfmt = strdup(argv[0] + strlen("--num-fmt="));
            if (!args->numfmt) { program_options_free(args); return errno; }
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
            err = parse_int64(&args->num_rows, argv[0], NULL, NULL);
            if (err) { program_options_free(args); return err; }
            args->num_columns = args->num_rows;
            args->num_nonzeros = args->num_rows*args->num_columns;
        } else if (num_positional_arguments_consumed == 1) {
            err = parse_int64(&args->num_columns, argv[0], NULL, NULL);
            if (err) { program_options_free(args); return err; }
            args->num_nonzeros = args->num_rows*args->num_columns;
        } else if (num_positional_arguments_consumed == 2) {
            err = parse_int64(&args->num_nonzeros, argv[0], NULL, NULL);
            if (err) { program_options_free(args); return err; }
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

static inline int64_t randrange(int64_t M, int64_t N)
{
    return M + rand() / (RAND_MAX / (N-M+1) + 1);
}

static int mtxfilerand(
    struct mtxfile * mtxfile,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t M,
    int64_t N)
{
    int err;
    union mtxfiledata * data = &mtxfile->data;
    if (object == mtxfile_matrix) {
        if (format == mtxfile_array) {
            err = mtxfile_alloc_matrix_array(
                mtxfile, field, symmetry, precision, num_rows, num_columns);
            if (err) return err;
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_real_single[k] = randrange(M,N);
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_real_double[k] = randrange(M,N);
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->array_complex_single[k][0] = randrange(M,N);
                        data->array_complex_single[k][1] = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->array_complex_double[k][0] = randrange(M,N);
                        data->array_complex_double[k][1] = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_integer_single[k] = randrange(M,N);
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_integer_double[k] = randrange(M,N);
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (format == mtxfile_coordinate) {
            err = mtxfile_alloc_matrix_coordinate(
                mtxfile, field, symmetry, precision,
                num_rows, num_columns, num_nonzeros);
            if (err) return err;
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->matrix_coordinate_real_single[k].i = randrange(1,num_rows);
                        data->matrix_coordinate_real_single[k].j = randrange(1,num_columns);
                        data->matrix_coordinate_real_single[k].a = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->matrix_coordinate_real_double[k].i = randrange(1,num_rows);
                        data->matrix_coordinate_real_double[k].j = randrange(1,num_columns);
                        data->matrix_coordinate_real_double[k].a = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->matrix_coordinate_complex_single[k].i = randrange(1,num_rows);
                        data->matrix_coordinate_complex_single[k].j = randrange(1,num_columns);
                        data->matrix_coordinate_complex_single[k].a[0] = randrange(M,N);
                        data->matrix_coordinate_complex_single[k].a[1] = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->matrix_coordinate_complex_double[k].i = randrange(1,num_rows);
                        data->matrix_coordinate_complex_double[k].j = randrange(1,num_columns);
                        data->matrix_coordinate_complex_double[k].a[0] = randrange(M,N);
                        data->matrix_coordinate_complex_double[k].a[1] = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->matrix_coordinate_integer_single[k].i = randrange(1,num_rows);
                        data->matrix_coordinate_integer_single[k].j = randrange(1,num_columns);
                        data->matrix_coordinate_integer_single[k].a = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->matrix_coordinate_integer_double[k].i = randrange(1,num_rows);
                        data->matrix_coordinate_integer_double[k].j = randrange(1,num_columns);
                        data->matrix_coordinate_integer_double[k].a = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < mtxfile->datasize; k++) {
                    data->matrix_coordinate_pattern[k].i = randrange(1,num_rows);
                    data->matrix_coordinate_pattern[k].j = randrange(1,num_columns);
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }

    } else if (object == mtxfile_vector) {
        if (format == mtxfile_array) {
            err = mtxfile_alloc_vector_array(mtxfile, field, precision, num_rows);
            if (err) return err;
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_real_single[k] = randrange(M,N);
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_real_double[k] = randrange(M,N);
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->array_complex_single[k][0] = randrange(M,N);
                        data->array_complex_single[k][1] = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->array_complex_double[k][0] = randrange(M,N);
                        data->array_complex_double[k][1] = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_integer_single[k] = randrange(M,N);
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++)
                        data->array_integer_double[k] = randrange(M,N);
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_FIELD; }
        } else if (format == mtxfile_coordinate) {
            err = mtxfile_alloc_vector_coordinate(
                mtxfile, field, precision, num_rows, num_nonzeros);
            if (err) return err;
            if (field == mtxfile_real) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->vector_coordinate_real_single[k].i = randrange(1,num_rows);
                        data->vector_coordinate_real_single[k].a = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->vector_coordinate_real_double[k].i = randrange(1,num_rows);
                        data->vector_coordinate_real_double[k].a = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->vector_coordinate_complex_single[k].i = randrange(1,num_rows);
                        data->vector_coordinate_complex_single[k].a[0] = randrange(M,N);
                        data->vector_coordinate_complex_single[k].a[1] = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->vector_coordinate_complex_double[k].i = randrange(1,num_rows);
                        data->vector_coordinate_complex_double[k].a[0] = randrange(M,N);
                        data->vector_coordinate_complex_double[k].a[1] = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->vector_coordinate_integer_single[k].i = randrange(1,num_rows);
                        data->vector_coordinate_integer_single[k].a = randrange(M,N);
                    }
                } else if (precision == mtx_double) {
                    for (int64_t k = 0; k < mtxfile->datasize; k++) {
                        data->vector_coordinate_integer_double[k].i = randrange(1,num_rows);
                        data->vector_coordinate_integer_double[k].a = randrange(M,N);
                    }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_pattern) {
                for (int64_t k = 0; k < mtxfile->datasize; k++)
                    data->vector_coordinate_pattern[k].i = randrange(1,num_rows);
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_FIELD; }
        } else { return MTX_ERR_INVALID_MTX_FORMAT; }
    } else { return MTX_ERR_INVALID_MTX_OBJECT; }
    return MTX_SUCCESS;
}

/**
 * ‘main()’.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
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

    /* seed the pseudo-random number generator */
    if (args.seed_given) { srand(args.seed); }
    else { srand(time(NULL)); }
    int64_t M = args.min <= args.max ? args.min : args.max;
    int64_t N = args.min <= args.max ? args.max : args.min;

    /* 2. generate a random matrix or vector */
    if (args.verbose > 0) {
        fprintf(stderr, "mtxfilerand: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    err = mtxfilerand(
        &mtxfile, args.object, args.format, args.field, args.symmetry, args.precision,
        args.num_rows, args.num_columns,
        args.object == mtxfile_matrix ? args.num_nonzeros : args.num_columns,
        M, N);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, mtxstrerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    mtxfilecomments_printf(
        &mtxfile.comments, "%% This file was generated by %s %s\n",
        program_name, program_version);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f Mnz/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * args.num_nonzeros / timespec_duration(t0, t1));
    }

    /* 3. write the matrix to standard output */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_fwrite: ");
            fflush(stderr);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
        int64_t bytes_written = 0;
        err = mtxfile_fwrite(&mtxfile, stdout, args.numfmt, &bytes_written);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, mtxstrerror(err));
            mtxfile_free(&mtxfile);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_written / timespec_duration(t0, t1));
        }
    }

    /* 4. Clean up. */
    mtxfile_free(&mtxfile);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
