/* This file is part of Libmtx.
 *
 * Copyright (C) 2023 James D. Trotter
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
 *
 * Last modified: 2023-02-09
 *
 * Generate random matrices and vectors in Matrix Market format.
 */

#include <libmtx/libmtx.h>

#include "parse.h"

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
    "Copyright (C) 2023 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

/**
 * ‘generator_type’ enumerates different kinds of random matrix
 * generators.
 */
enum generator_type
{
    generator_uniform, /* uniformly random matrices */
    generator_blkdiag, /* block-diagonal matrices */
};

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    /* options for matrix type */
    enum mtxfileobject object;
    enum mtxfileformat format;
    enum mtxfilefield field;
    enum mtxfilesymmetry symmetry;

    /* options for matrix size */
    int64_t num_rows;
    int64_t num_columns;

    /* options for matrix generators */
    enum generator_type generator;
    int64_t rowoffset;
    int64_t coloffset;

    /* options for uniformly random generator */
    int64_t num_nonzeros;

    /* options for block diagonal generator */
    int64_t nblocks;
    int64_t nblockrows;
    int64_t nblockcols;
    int64_t nblocknonzeros;

    /* options for random matrix values */
    int64_t min;
    int64_t max;

    /* random seeding */
    bool seed_given;
    int seed;

    /* output options */
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

    /* options for matrix size */
    args->num_rows = 10;
    args->num_columns = 10;

    args->generator = generator_uniform;
    args->rowoffset = 1;
    args->coloffset = 1;

    /* options for uniformly random generator */
    args->num_nonzeros = 10;

    /* options for block diagonal generator */
    args->nblocks = 2;
    args->nblockrows = 5;
    args->nblockcols = 5;
    args->nblocknonzeros = 0;

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
    fprintf(f, "Usage: %s [OPTION..]\n", program_name);
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
    fprintf(f, " Options for matrix type and size are:\n");
    fprintf(f, "  --object OBJECT      matrix or vector. [matrix]\n");
    fprintf(f, "  --format FORMAT      array or coordinate. [coordinate]\n");
    fprintf(f, "  --field FIELD        real, integer, complex or pattern. [real]\n");
    fprintf(f, "  --symmetry SYMMETRY  general, symmetric, skew-symmetric or hermitian. [general]\n");
    fprintf(f, "  --rows N             number of rows. [10]\n");
    fprintf(f, "  --cols N             number of columns. [10]\n");
    fprintf(f, "\n");

    fprintf(f, " Options for matrix generators are:\n");
    fprintf(f, "  --generator TYPE     uniform or block-diagonal. [uniform]\n");
    fprintf(f, "  --row-offset N       offset to first nonzero row. [1]\n");
    fprintf(f, "  --col-offset N       offset to first nonzero column. [1]\n");
    fprintf(f, "\n");

    fprintf(f, " Options for uniformly random generator are:\n");
    fprintf(f, "  --nonzeros N         number of nonzeros. [10]\n");
    fprintf(f, "\n");

    fprintf(f, " Options for block diagonal generator are:\n");
    fprintf(f, "  --blocks N           number of blocks. [2]\n");
    fprintf(f, "  --block-rows N       number of rows in each block. [5]\n");
    fprintf(f, "  --block-cols N       number of columns in each block. [5]\n");
    fprintf(f, "  --block-nonzeros N   If set, each block is sparse with N nonzeros,\n");
    fprintf(f, "                       otherwise blocks are dense by default. [0]\n");
    fprintf(f, "\n");

    fprintf(f, " Other options are:\n");
    fprintf(f, "  --precision PRECISION  precision for numerical values: single or double. [double]\n");
    fprintf(f, "  --min N                lower range for randomly generated numerical values [0]\n");
    fprintf(f, "  --max N                upper range for randomly generated numerical values [%d]\n", RAND_MAX);
    fprintf(f, "  --seed N               seed for pseudo-random number generator.\n");
    fprintf(f, "                         By default, the generator is seeded with the current time.\n");
    fprintf(f, "\n");
    fprintf(f, "  --num-fmt FMT          Format string for outputting numerical values.\n");
    fprintf(f, "                         For real, double and complex values, the format specifiers\n");
    fprintf(f, "                         '%%e', '%%E', '%%f', '%%F', '%%g' or '%%G' may be used,\n");
    fprintf(f, "                         whereas '%%d' must be used for integers. Flags, field width\n");
    fprintf(f, "                         and precision can optionally be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  -q, --quiet            do not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose          be more verbose\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help             display this help and exit\n");
    fprintf(f, "  --version              display version information and exit\n");
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
        if (strstr(argv[0], "--object") == argv[0]) {
            int n = strlen("--object");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxfileobject(&args->object, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--format") == argv[0]) {
            int n = strlen("--format");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxfileformat(&args->format, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--field") == argv[0]) {
            int n = strlen("--field");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxfilefield(&args->field, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--symmetry") == argv[0]) {
            int n = strlen("--symmetry");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxfilesymmetry(&args->symmetry, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--precision") == argv[0]) {
            int n = strlen("--precision");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_mtxprecision(&args->precision, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        /* options for matrix size */
        if (strstr(argv[0], "--rows") == argv[0]) {
            int n = strlen("--rows");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->num_rows, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--cols") == argv[0]) {
            int n = strlen("--cols");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->num_columns, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        /* options for matrix generators */
        if (strstr(argv[0], "--generator") == argv[0]) {
            int n = strlen("--generator");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { s = NULL; }
            if (s) {
                if (strcmp(s, "uniform") == 0) {
                    args->generator = generator_uniform;
                } else if (strcmp(s, "block-diagonal") == 0) {
                    args->generator = generator_blkdiag;
                } else { program_options_free(args); return EINVAL; }
                (*nargs)++; argv++; continue;
            }
        }
        if (strstr(argv[0], "--row-offset") == argv[0]) {
            int n = strlen("--row-offset");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->rowoffset, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--col-offset") == argv[0]) {
            int n = strlen("--col-offset");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->coloffset, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        /* options for uniformly random generator */
        if (strstr(argv[0], "--nonzeros") == argv[0]) {
            int n = strlen("--nonzeros");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->num_nonzeros, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        /* options for block diagonal generator */
        if (strstr(argv[0], "--blocks") == argv[0]) {
            int n = strlen("--blocks");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->nblocks, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--block-rows") == argv[0]) {
            int n = strlen("--block-rows");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->nblockrows, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--block-cols") == argv[0]) {
            int n = strlen("--block-cols");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->nblockcols, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--block-nonzeros") == argv[0]) {
            int n = strlen("--block-nonzeros");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->nblocknonzeros, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--min") == argv[0]) {
            int n = strlen("--min");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->min, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--max") == argv[0]) {
            int n = strlen("--max");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int64(&args->max, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--seed") == argv[0]) {
            int n = strlen("--seed");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->seed, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
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

        /* no positional arguments */
        if (num_positional_arguments_consumed == 0) {
            program_options_free(args); return EINVAL;
        } else { program_options_free(args); return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
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

/**
 * ‘mtxrand_uniform()’ generates a uniformly random sparse matrix.
 */
static int mtxrand_uniform(
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
                    if (symmetry == mtxfile_general ||
                        symmetry == mtxfile_symmetric ||
                        symmetry == mtxfile_skew_symmetric)
                    {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->array_complex_single[k][0] = randrange(M,N);
                            data->array_complex_single[k][1] = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_hermitian) {
                        for (int64_t i = 0, k = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < i; j++, k++) {
                                data->array_complex_single[k][0] = randrange(M,N);
                                data->array_complex_single[k][1] = randrange(M,N);
                            }
                            data->array_complex_single[k][0] = randrange(M,N);
                            data->array_complex_single[k][1] = 0;
                            k++;
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
                } else if (precision == mtx_double) {
                    if (symmetry == mtxfile_general ||
                        symmetry == mtxfile_symmetric ||
                        symmetry == mtxfile_skew_symmetric)
                    {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->array_complex_double[k][0] = randrange(M,N);
                            data->array_complex_double[k][1] = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_hermitian) {
                        for (int64_t i = 0, k = 0; i < num_rows; i++) {
                            for (int64_t j = 0; j < i; j++, k++) {
                                data->array_complex_double[k][0] = randrange(M,N);
                                data->array_complex_double[k][1] = randrange(M,N);
                            }
                            data->array_complex_double[k][0] = randrange(M,N);
                            data->array_complex_double[k][1] = 0;
                            k++;
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
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
                    if (symmetry == mtxfile_general ||
                        symmetry == mtxfile_symmetric ||
                        symmetry == mtxfile_hermitian)
                    {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_real_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_real_single[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_real_single[k].a = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_skew_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_real_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_real_single[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_real_single[k].a =
                                data->matrix_coordinate_real_single[k].i == data->matrix_coordinate_real_single[k].j ? 0 : randrange(M,N);
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
                } else if (precision == mtx_double) {
                    if (symmetry == mtxfile_general ||
                        symmetry == mtxfile_symmetric ||
                        symmetry == mtxfile_hermitian)
                    {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_real_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_real_double[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_real_double[k].a = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_skew_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_real_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_real_double[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_real_double[k].a =
                                data->matrix_coordinate_real_double[k].i == data->matrix_coordinate_real_double[k].j ? 0 : randrange(M,N);
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_complex) {
                if (precision == mtx_single) {
                    if (symmetry == mtxfile_general || symmetry == mtxfile_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_complex_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_complex_single[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_complex_single[k].a[0] = randrange(M,N);
                            data->matrix_coordinate_complex_single[k].a[1] = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_skew_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_complex_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_complex_single[k].j = randrange(1,num_columns);
                            if (data->matrix_coordinate_complex_single[k].i != data->matrix_coordinate_complex_single[k].j) {
                                data->matrix_coordinate_complex_single[k].a[0] = randrange(M,N);
                                data->matrix_coordinate_complex_single[k].a[1] = randrange(M,N);
                            } else {
                                data->matrix_coordinate_complex_single[k].a[0] = 0;
                                data->matrix_coordinate_complex_single[k].a[1] = 0;
                            }
                        }
                    } else if (symmetry == mtxfile_hermitian) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_complex_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_complex_single[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_complex_single[k].a[0] = randrange(M,N);
                            if (data->matrix_coordinate_complex_single[k].i != data->matrix_coordinate_complex_single[k].j) {
                                data->matrix_coordinate_complex_single[k].a[1] = randrange(M,N);
                            } else {
                                data->matrix_coordinate_complex_single[k].a[1] = 0;
                            }
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
                } else if (precision == mtx_double) {
                    if (symmetry == mtxfile_general || symmetry == mtxfile_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_complex_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_complex_double[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_complex_double[k].a[0] = randrange(M,N);
                            data->matrix_coordinate_complex_double[k].a[1] = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_skew_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_complex_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_complex_double[k].j = randrange(1,num_columns);
                            if (data->matrix_coordinate_complex_double[k].i != data->matrix_coordinate_complex_double[k].j) {
                                data->matrix_coordinate_complex_double[k].a[0] = randrange(M,N);
                                data->matrix_coordinate_complex_double[k].a[1] = randrange(M,N);
                            } else {
                                data->matrix_coordinate_complex_double[k].a[0] = 0;
                                data->matrix_coordinate_complex_double[k].a[1] = 0;
                            }
                        }
                    } else if (symmetry == mtxfile_hermitian) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_complex_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_complex_double[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_complex_double[k].a[0] = randrange(M,N);
                            if (data->matrix_coordinate_complex_double[k].i != data->matrix_coordinate_complex_double[k].j) {
                                data->matrix_coordinate_complex_double[k].a[1] = randrange(M,N);
                            } else {
                                data->matrix_coordinate_complex_double[k].a[1] = 0;
                            }
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
                } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
            } else if (field == mtxfile_integer) {
                if (precision == mtx_single) {
                    if (symmetry == mtxfile_general ||
                        symmetry == mtxfile_symmetric ||
                        symmetry == mtxfile_hermitian)
                    {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_integer_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_integer_single[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_integer_single[k].a = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_skew_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_integer_single[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_integer_single[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_integer_single[k].a =
                                data->matrix_coordinate_integer_single[k].i == data->matrix_coordinate_integer_single[k].j ? 0 : randrange(M,N);
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
                } else if (precision == mtx_double) {
                    if (symmetry == mtxfile_general ||
                        symmetry == mtxfile_symmetric ||
                        symmetry == mtxfile_hermitian)
                    {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_integer_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_integer_double[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_integer_double[k].a = randrange(M,N);
                        }
                    } else if (symmetry == mtxfile_skew_symmetric) {
                        for (int64_t k = 0; k < mtxfile->datasize; k++) {
                            data->matrix_coordinate_integer_double[k].i = randrange(1,num_rows);
                            data->matrix_coordinate_integer_double[k].j = randrange(1,num_columns);
                            data->matrix_coordinate_integer_double[k].a =
                                data->matrix_coordinate_integer_double[k].i == data->matrix_coordinate_integer_double[k].j ? 0 : randrange(M,N);
                        }
                    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
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
 * ‘mtxrand_blkdiag()’ generates a random, block-diagonal sparse
 * matrix.
 */
static int mtxrand_blkdiag(
    struct mtxfile * mtxfile,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    int64_t num_rows,
    int64_t num_columns,
    int64_t rowoffset,
    int64_t coloffset,
    int64_t nblocks,
    int64_t nblockrows,
    int64_t nblockcols,
    int64_t nblocknonzeros,
    int64_t M,
    int64_t N)
{
    int err;
    union mtxfiledata * data = &mtxfile->data;
    if (object != mtxfile_matrix) return MTX_ERR_NOT_SUPPORTED;
    if (format != mtxfile_coordinate) return MTX_ERR_NOT_SUPPORTED;
    if (rowoffset+nblocks*nblockrows-1 > num_rows) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    if (coloffset+nblocks*nblockcols-1 > num_columns) return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    int64_t num_nonzeros = nblocks * (nblocknonzeros > 0 ? nblocknonzeros : nblockrows*nblockcols);
    err = mtxfile_alloc_matrix_coordinate(
        mtxfile, field, symmetry, precision,
        num_rows, num_columns, num_nonzeros);
    if (err) return err;
    if (field == mtxfile_real) {
        if (precision == mtx_single) {
            if (symmetry == mtxfile_general ||
                symmetry == mtxfile_symmetric ||
                symmetry == mtxfile_hermitian)
            {
                int64_t i = rowoffset, j = coloffset, k = 0;
                for (int64_t b = 0; b < nblocks; b++) {
                    if (nblocknonzeros > 0) {
                        for (int64_t l = 0; l < nblocknonzeros; k++, l++) {
                            data->matrix_coordinate_real_single[k].i = i+randrange(1,nblockrows);
                            data->matrix_coordinate_real_single[k].j = j+randrange(1,nblockcols);
                            data->matrix_coordinate_real_single[k].a = randrange(M,N);
                        }
                    } else {
                        for (int64_t m = 0; m < nblockrows; m++) {
                            for (int64_t n = 0; n < nblockcols; n++, k++) {
                                data->matrix_coordinate_real_single[k].i = i+m;
                                data->matrix_coordinate_real_single[k].j = j+n;
                                data->matrix_coordinate_real_single[k].a = randrange(M,N);
                            }
                        }
                    }
                    i += nblockrows;
                    j += nblockcols;
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
        } else if (precision == mtx_double) {
            if (symmetry == mtxfile_general ||
                symmetry == mtxfile_symmetric ||
                symmetry == mtxfile_hermitian)
            {
                int64_t i = rowoffset, j = coloffset, k = 0;
                for (int64_t b = 0; b < nblocks; b++) {
                    if (nblocknonzeros > 0) {
                        for (int64_t l = 0; l < nblocknonzeros; k++, l++) {
                            data->matrix_coordinate_real_double[k].i = i+randrange(1,nblockrows);
                            data->matrix_coordinate_real_double[k].j = j+randrange(1,nblockcols);
                            data->matrix_coordinate_real_double[k].a = randrange(M,N);
                        }
                    } else {
                        for (int64_t m = 0; m < nblockrows; m++) {
                            for (int64_t n = 0; n < nblockcols; n++, k++) {
                                data->matrix_coordinate_real_double[k].i = i+m;
                                data->matrix_coordinate_real_double[k].j = j+n;
                                data->matrix_coordinate_real_double[k].a = randrange(M,N);
                            }
                        }
                    }
                    i += nblockrows;
                    j += nblockcols;
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
        } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_integer) {
        if (precision == mtx_single) {
            if (symmetry == mtxfile_general ||
                symmetry == mtxfile_symmetric ||
                symmetry == mtxfile_hermitian)
            {
                int64_t i = rowoffset, j = coloffset, k = 0;
                for (int64_t b = 0; b < nblocks; b++) {
                    if (nblocknonzeros > 0) {
                        for (int64_t l = 0; l < nblocknonzeros; k++, l++) {
                            data->matrix_coordinate_integer_single[k].i = i+randrange(1,nblockrows);
                            data->matrix_coordinate_integer_single[k].j = j+randrange(1,nblockcols);
                            data->matrix_coordinate_integer_single[k].a = randrange(M,N);
                        }
                    } else {
                        for (int64_t m = 0; m < nblockrows; m++) {
                            for (int64_t n = 0; n < nblockcols; n++, k++) {
                                data->matrix_coordinate_integer_single[k].i = i+m;
                                data->matrix_coordinate_integer_single[k].j = j+n;
                                data->matrix_coordinate_integer_single[k].a = randrange(M,N);
                            }
                        }
                    }
                    i += nblockrows;
                    j += nblockcols;
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
        } else if (precision == mtx_double) {
            if (symmetry == mtxfile_general ||
                symmetry == mtxfile_symmetric ||
                symmetry == mtxfile_hermitian)
            {
                int64_t i = rowoffset, j = coloffset, k = 0;
                for (int64_t b = 0; b < nblocks; b++) {
                    if (nblocknonzeros > 0) {
                        for (int64_t l = 0; l < nblocknonzeros; k++, l++) {
                            data->matrix_coordinate_integer_double[k].i = i+randrange(1,nblockrows);
                            data->matrix_coordinate_integer_double[k].j = j+randrange(1,nblockcols);
                            data->matrix_coordinate_integer_double[k].a = randrange(M,N);
                        }
                    } else {
                        for (int64_t m = 0; m < nblockrows; m++) {
                            for (int64_t n = 0; n < nblockcols; n++, k++) {
                                data->matrix_coordinate_integer_double[k].i = i+m;
                                data->matrix_coordinate_integer_double[k].j = j+n;
                                data->matrix_coordinate_integer_double[k].a = randrange(M,N);
                            }
                        }
                    }
                    i += nblockrows;
                    j += nblockcols;
                }
            } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_SYMMETRY; }
        } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_PRECISION; }
    } else if (field == mtxfile_pattern) {
        int64_t i = rowoffset, j = coloffset, k = 0;
        for (int64_t b = 0; b < nblocks; b++) {
            if (nblocknonzeros > 0) {
                for (int64_t l = 0; l < nblocknonzeros; k++, l++) {
                    data->matrix_coordinate_pattern[k].i = i+randrange(1,nblockrows);
                    data->matrix_coordinate_pattern[k].j = j+randrange(1,nblockcols);
                }
            } else {
                for (int64_t m = 0; m < nblockrows; m++) {
                    for (int64_t n = 0; n < nblockcols; n++, k++) {
                        data->matrix_coordinate_pattern[k].i = i+m;
                        data->matrix_coordinate_pattern[k].j = j+n;
                    }
                }
            }
            i += nblockrows;
            j += nblockcols;
        }
    } else { mtxfile_free(mtxfile); return MTX_ERR_INVALID_MTX_FIELD; }
    return MTX_SUCCESS;
}

/**
 * ‘mtxrand()’ generates a random sparse matrix of the designated
 * type.
 */
static int mtxrand(
    struct mtxfile * mtxfile,
    enum mtxfileobject object,
    enum mtxfileformat format,
    enum mtxfilefield field,
    enum mtxfilesymmetry symmetry,
    enum mtxprecision precision,
    enum generator_type generator,
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t rowoffset,
    int64_t coloffset,
    int64_t nblocks,
    int64_t nblockrows,
    int64_t nblockcols,
    int64_t nblocknonzeros,
    int64_t amin,
    int64_t amax)
{
    if (generator == generator_uniform) {
        return mtxrand_uniform(
            mtxfile, object, format, field, symmetry, precision,
            num_rows, num_columns,
            object == mtxfile_matrix ? num_nonzeros : num_columns,
            amin, amax);
    } else if (generator == generator_blkdiag) {
        return mtxrand_blkdiag(
            mtxfile, object, format, field, symmetry, precision,
            num_rows, num_columns, rowoffset, coloffset,
            nblocks, nblockrows, nblockcols, nblocknonzeros,
            amin, amax);
    } else { return MTX_ERR_NOT_SUPPORTED; }
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
        fprintf(stderr, "mtxrand: ");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    err = mtxrand(
        &mtxfile,
        args.object, args.format, args.field, args.symmetry, args.precision,
        args.generator, args.num_rows, args.num_columns, args.num_nonzeros,
        args.rowoffset, args.coloffset,
        args.nblocks, args.nblockrows, args.nblockcols, args.nblocknonzeros,
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
