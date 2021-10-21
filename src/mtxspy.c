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
 * Last modified: 2021-08-09
 *
 * Draw an image of a matrix sparsity pattern and save to a PNG file.
 */

#include <libmtx/libmtx.h>

#include "../libmtx/util/parse.h"

#include <png.h>

#include <errno.h>

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "mtxspy";
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
    enum mtx_precision precision;
    bool gzip;
    char * png_output_path;
    int max_width;
    int max_height;
    double gamma;
    int fgcolor;
    int bgcolor;
    char * title;
    char * author;
    char * description;
    char * copyright;
    char * email;
    char * url;
    int verbose;
};

/**
 * `program_options_init()` configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->mtx_path = NULL;
    args->precision = mtx_double;
    args->gzip = false;
    args->png_output_path = strdup("out.png");
    args->max_width = 1000;
    args->max_height = 1000;
    args->gamma = -1.0;
    args->fgcolor = 0x000000;
    args->bgcolor = 0xFFFFFF;
    args->title = NULL;
    args->author = NULL;
    args->description = NULL;
    args->copyright = NULL;
    args->email = NULL;
    args->url = NULL;
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
    if (args->png_output_path)
        free(args->png_output_path);
    if(args->title)
        free(args->title);
    if(args->author)
        free(args->author);
    if(args->description)
        free(args->description);
    if(args->copyright)
        free(args->copyright);
    if(args->email)
        free(args->email);
    if(args->url)
        free(args->url);
}

/**
 * `program_options_print_help()` prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Save an image of a matrix sparsity pattern to a PNG file.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --precision=PRECISION\tprecision used to represent matrix or\n");
    fprintf(f, "\t\t\tvector values: single or double. (default: double)\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip\tfilter files through gzip\n");
    fprintf(f, "  --output-path=FILE\tpath for outputting the image (default: out.png).\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
    fprintf(f, "  --max-height=M\tmaximum height of image in pixels (default: 1000).\n");
    fprintf(f, "  --max-width=N\t\tmaximum width of image in pixels (default: 1000).\n");
    fprintf(f, "  --gamma=GAMMA\t\tgamma value (default: none)\n");
    fprintf(f, "  --fgcolor=COLOR\tforeground color in hexadecimal (default: #000000).\n");
    fprintf(f, "  --bgcolor=COLOR\tbackground color in hexadecimal (default: #FFFFFF).\n");
    fprintf(f, "\n");
    fprintf(f, "  --title=TEXT\t\tTitle field of the PNG image.\n");
    fprintf(f, "  --author=TEXT\t\tAuthor field of the PNG image.\n");
    fprintf(f, "  --description=TEXT\tDescription field of the PNG image.\n");
    fprintf(f, "  --copyright=TEXT\tCopyright field of the PNG image.\n");
    fprintf(f, "  --email=TEXT\t\tE-mail field of the PNG image.\n");
    fprintf(f, "  --url=TEXT\t\tURL field of the PNG image.\n");
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

        char * s;

        if (strcmp((*argv)[0], "--precision") == 0) {
            if (*argc < 2) {
                program_options_free(args);
                return EINVAL;
            }
            char * s = (*argv)[1];
            err = mtx_precision_parse(&args->precision, NULL, NULL, s, "");
            if (err) {
                program_options_free(args);
                return EINVAL;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--precision=") == (*argv)[0]) {
            char * s = (*argv)[0] + strlen("--precision=");
            err = mtx_precision_parse(&args->precision, NULL, NULL, s, "");
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

        /* Parse output path. */
        if (strcmp((*argv)[0], "--output-path") == 0 && *argc >= 2) {
            if (args->png_output_path)
                free(args->png_output_path);
            args->png_output_path = strdup((*argv)[1]);
            if (!args->png_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--output-path=") == (*argv)[0]) {
            if (args->png_output_path)
                free(args->png_output_path);
            args->png_output_path =
                strdup((*argv)[0] + strlen("--output-path="));
            if (!args->png_output_path) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        /* Parse image maximum width and height. */
        if (strcmp((*argv)[0], "--max-height") == 0 && *argc >= 2) {
            err = parse_int32((*argv)[1], NULL, &args->max_height, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--max-height=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--max-height="), NULL,
                &args->max_height, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }
        if (strcmp((*argv)[0], "--max-width") == 0 && *argc >= 2) {
            err = parse_int32((*argv)[1], NULL, &args->max_width, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--max-width=") == (*argv)[0]) {
            err = parse_int32(
                (*argv)[0] + strlen("--max-width="), NULL,
                &args->max_width, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--fgcolor") == 0 && *argc >= 2) {
            if ((*argv)[1][0] == '#') // Skip the optional '#' character
                ((*argv)[1])++;
            err = parse_int32_hex((*argv)[1], NULL, &args->fgcolor, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--fgcolor=") == (*argv)[0]) {
            int n = strlen("--fgcolor=");
            if ((*argv)[0][n] == '#') // Skip the optional '#' character
                n++;
            err = parse_int32_hex(
                (*argv)[0] + n, NULL, &args->fgcolor, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }
        if (strcmp((*argv)[0], "--bgcolor") == 0 && *argc >= 2) {
            if ((*argv)[1][0] == '#') // Skip the optional '#' character
                ((*argv)[1])++;
            err = parse_int32_hex((*argv)[1], NULL, &args->bgcolor, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--bgcolor=") == (*argv)[0]) {
            int n = strlen("--bgcolor=");
            if ((*argv)[0][n] == '#') // Skip the optional '#' character
                n++;
            err = parse_int32_hex(
                (*argv)[0] + n, NULL, &args->bgcolor, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--gamma") == 0 && *argc >= 2) {
            err = parse_double((*argv)[1], NULL, &args->gamma, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--gamma=") == (*argv)[0]) {
            err = parse_double(
                (*argv)[0] + strlen("--gamma="), NULL,
                &args->gamma, NULL);
            if (err) {
                program_options_free(args);
                return err;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--title") == 0 && *argc >= 2) {
            if (args->title)
                free(args->title);
            args->title = strdup((*argv)[1]);
            if (!args->title) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--title=") == (*argv)[0]) {
            if (args->title)
                free(args->title);
            args->title = strdup((*argv)[0] + strlen("--title="));
            if (!args->title) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--author") == 0 && *argc >= 2) {
            if (args->author)
                free(args->author);
            args->author = strdup((*argv)[1]);
            if (!args->author) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--author=") == (*argv)[0]) {
            if (args->author)
                free(args->author);
            args->author = strdup((*argv)[0] + strlen("--author="));
            if (!args->author) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--description") == 0 && *argc >= 2) {
            if (args->description)
                free(args->description);
            args->description = strdup((*argv)[1]);
            if (!args->description) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--description=") == (*argv)[0]) {
            if (args->description)
                free(args->description);
            args->description = strdup((*argv)[0] + strlen("--description="));
            if (!args->description) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--copyright") == 0 && *argc >= 2) {
            if (args->copyright)
                free(args->copyright);
            args->copyright = strdup((*argv)[1]);
            if (!args->copyright) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--copyright=") == (*argv)[0]) {
            if (args->copyright)
                free(args->copyright);
            args->copyright = strdup((*argv)[0] + strlen("--copyright="));
            if (!args->copyright) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--email") == 0 && *argc >= 2) {
            if (args->email)
                free(args->email);
            args->email = strdup((*argv)[1]);
            if (!args->email) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--email=") == (*argv)[0]) {
            if (args->email)
                free(args->email);
            args->email = strdup((*argv)[0] + strlen("--email="));
            if (!args->email) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed++;
            continue;
        }

        if (strcmp((*argv)[0], "--url") == 0 && *argc >= 2) {
            if (args->url)
                free(args->url);
            args->url = strdup((*argv)[1]);
            if (!args->url) {
                program_options_free(args);
                return errno;
            }
            num_arguments_consumed += 2;
            continue;
        } else if (strstr((*argv)[0], "--url=") == (*argv)[0]) {
            if (args->url)
                free(args->url);
            args->url = strdup((*argv)[0] + strlen("--url="));
            if (!args->url) {
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

static void draw_sparsity_pattern_point(
    int width,
    int height,
    unsigned char * imgbuf,
    int num_rows,
    int num_columns,
    int i,
    int j,
    unsigned char r,
    unsigned char g,
    unsigned char b)
{
    double x = (((double) j + 0.5) / (double) num_columns) * (double) width;
    double y = (((double) i + 0.5) / (double) num_rows) * (double) height;
    int left = (int) floor(x);
    int right = (int) ceil(x);
    int top = (int) floor(y);
    int bottom = (int) ceil(y);
    for (int u = left; u < right; u++) {
        for (int v = top; v < bottom; v++) {
            int64_t w = (int64_t) v * (int64_t) width + (int64_t) u;
            imgbuf[w*3+0] = r;
            imgbuf[w*3+1] = g;
            imgbuf[w*3+2] = b;
        }
    }
}

/**
 * `draw_sparsity_pattern_binary()' renders the sparsity pattern of a
 * matrix to a 2D array of RGB values.
 */
static int draw_sparsity_pattern(
    const struct mtxfile * mtxfile,
    int fgcolor,
    int bgcolor,
    int max_width,
    int max_height,
    int * out_width,
    int * out_height,
    int * out_bit_depth,
    int * out_pixel_size,
    int * out_color_type,
    unsigned char ** out_imgbuf,
    unsigned char *** out_row_pointers)
{
    if (mtxfile->header.object != mtxfile_matrix)
        return MTX_ERR_INVALID_MTX_OBJECT;
    if (mtxfile->header.format != mtxfile_coordinate)
        return MTX_ERR_INVALID_MTX_FORMAT;

    int num_rows = mtxfile->size.num_rows;
    int num_columns = mtxfile->size.num_columns;
    int64_t num_nonzeros = mtxfile->size.num_nonzeros;
    int width;
    int height;
    if (num_rows < max_height && num_columns < max_width) {
        height = num_rows;
        width = num_columns;
    } else if (num_rows < max_height) {
        height = num_rows;
        width = max_width;
    } else if (num_columns < max_width) {
        height = max_height;
        width = num_columns;
    } else {
        if (num_rows == num_columns) {
            height = max_height;
            width = max_width;
        } else if (num_rows > num_columns) {
            height = max_height;
            width = ((int64_t) max_height * (int64_t) num_columns) / num_rows;
        } else {
            height = ((int64_t) max_width * (int64_t) num_rows) / num_columns;
            width = max_width;
        }
    }

    int pixel_size = 3 * sizeof(unsigned char);
    int bit_depth = CHAR_BIT;
    int color_type = PNG_COLOR_TYPE_RGB;
    int64_t imgbufsize = (int64_t) width * (int64_t) height * (int64_t) pixel_size;
    unsigned char * imgbuf = malloc(imgbufsize);
    if (!imgbuf)
        return MTX_ERR_ERRNO;

    unsigned char bg_red = ((bgcolor >> 16) & 0xFF);
    unsigned char bg_green = ((bgcolor >> 8) & 0xFF);
    unsigned char bg_blue = bgcolor & 0xFF;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int64_t k = (int64_t) i * (int64_t) width + (int64_t) j;
            imgbuf[k*3+0] = bg_red;
            imgbuf[k*3+1] = bg_green;
            imgbuf[k*3+2] = bg_blue;
        }
    }

    unsigned char ** row_pointers = malloc(sizeof(unsigned char *) * height);
    if (!row_pointers) {
        free(imgbuf);
        return MTX_ERR_ERRNO;
    }
    for (int i = 0; i < height; i++)
        row_pointers[i] = imgbuf + (int64_t) i * (int64_t) width * (int64_t) pixel_size;

    unsigned char fg_red = ((fgcolor >> 16) & 0xFF);
    unsigned char fg_green = ((fgcolor >> 8) & 0xFF);
    unsigned char fg_blue = fgcolor & 0xFF;

    if (mtxfile->header.field == mtxfile_real) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_real_single * data =
                mtxfile->data.matrix_coordinate_real_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
            }
            if (mtxfile->header.symmetry == mtxfile_symmetric ||
                mtxfile->header.symmetry == mtxfile_skew_symmetric)
            {
                for (int64_t k = 0; k < num_nonzeros; k++) {
                    draw_sparsity_pattern_point(
                        width, height, imgbuf, num_rows, num_columns,
                        data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
                }
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_real_double * data =
                mtxfile->data.matrix_coordinate_real_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
            }
            if (mtxfile->header.symmetry == mtxfile_symmetric ||
                mtxfile->header.symmetry == mtxfile_skew_symmetric)
            {
                for (int64_t k = 0; k < num_nonzeros; k++) {
                    draw_sparsity_pattern_point(
                        width, height, imgbuf, num_rows, num_columns,
                        data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
                }
            }
        } else {
            free(row_pointers);
            free(imgbuf);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_complex) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_complex_single * data =
                mtxfile->data.matrix_coordinate_complex_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
            }
            if (mtxfile->header.symmetry == mtxfile_symmetric ||
                mtxfile->header.symmetry == mtxfile_skew_symmetric ||
                mtxfile->header.symmetry == mtxfile_hermitian)
            {
                for (int64_t k = 0; k < num_nonzeros; k++) {
                    draw_sparsity_pattern_point(
                        width, height, imgbuf, num_rows, num_columns,
                        data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
                }
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_complex_double * data =
                mtxfile->data.matrix_coordinate_complex_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
            }
            if (mtxfile->header.symmetry == mtxfile_symmetric ||
                mtxfile->header.symmetry == mtxfile_skew_symmetric ||
                mtxfile->header.symmetry == mtxfile_hermitian)
            {
                for (int64_t k = 0; k < num_nonzeros; k++) {
                    draw_sparsity_pattern_point(
                        width, height, imgbuf, num_rows, num_columns,
                        data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
                }
            }
        } else {
            free(row_pointers);
            free(imgbuf);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_integer) {
        if (mtxfile->precision == mtx_single) {
            const struct mtxfile_matrix_coordinate_integer_single * data =
                mtxfile->data.matrix_coordinate_integer_single;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
            }
            if (mtxfile->header.symmetry == mtxfile_symmetric ||
                mtxfile->header.symmetry == mtxfile_skew_symmetric)
            {
                for (int64_t k = 0; k < num_nonzeros; k++) {
                    draw_sparsity_pattern_point(
                        width, height, imgbuf, num_rows, num_columns,
                        data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
                }
            }
        } else if (mtxfile->precision == mtx_double) {
            const struct mtxfile_matrix_coordinate_integer_double * data =
                mtxfile->data.matrix_coordinate_integer_double;
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
            }
            if (mtxfile->header.symmetry == mtxfile_symmetric ||
                mtxfile->header.symmetry == mtxfile_skew_symmetric)
            {
                for (int64_t k = 0; k < num_nonzeros; k++) {
                    draw_sparsity_pattern_point(
                        width, height, imgbuf, num_rows, num_columns,
                        data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
                }
            }
        } else {
            free(row_pointers);
            free(imgbuf);
            return MTX_ERR_INVALID_PRECISION;
        }
    } else if (mtxfile->header.field == mtxfile_pattern) {
        const struct mtxfile_matrix_coordinate_pattern * data =
            mtxfile->data.matrix_coordinate_pattern;
        for (int64_t k = 0; k < num_nonzeros; k++) {
            draw_sparsity_pattern_point(
                width, height, imgbuf, num_rows, num_columns,
                data[k].i-1, data[k].j-1, fg_red, fg_green, fg_blue);
        }
        if (mtxfile->header.symmetry == mtxfile_symmetric ||
            mtxfile->header.symmetry == mtxfile_skew_symmetric)
        {
            for (int64_t k = 0; k < num_nonzeros; k++) {
                draw_sparsity_pattern_point(
                    width, height, imgbuf, num_rows, num_columns,
                    data[k].j-1, data[k].i-1, fg_red, fg_green, fg_blue);
            }
        }
    } else {
        free(row_pointers);
        free(imgbuf);
        return MTX_ERR_INVALID_MTX_FIELD;
    }

    *out_width = width;
    *out_height = height;
    *out_bit_depth = bit_depth;
    *out_pixel_size = pixel_size;
    *out_color_type = color_type;
    *out_imgbuf = imgbuf;
    *out_row_pointers = row_pointers;
    return MTX_SUCCESS;
}

struct png_error
{
    jmp_buf jmpbuf;
    char * description;
};

/**
 * `png_handle_error()' is an error-handling callback function for
 * errors from libpng.
 */
static void png_handle_error(
    png_structp png_ptr,
    png_const_charp error_msg)
{
    png_voidp error_ptr = png_get_error_ptr(png_ptr);
    struct png_error * err = (struct png_error *) error_ptr;
    if (!err) {
        fprintf(stderr, "%s: libpng error: %s\n",
                program_invocation_short_name, error_msg);
        exit(EXIT_FAILURE);
    }

    err->description = strdup(error_msg);
    longjmp(err->jmpbuf, 1);
}

/**
 * `png_handle_warning()' is an error-handling callback function for
 * warnings from libpng.
 */
static void png_handle_warning(
    png_structp png_ptr,
    png_const_charp warning_msg)
{
    png_voidp error_ptr = png_get_error_ptr(png_ptr);
    struct png_error * err = (struct png_error *) error_ptr;
    if (!err) {
        fprintf(stderr, "%s: libpng warning: %s\n",
                program_invocation_short_name, warning_msg);
        exit(EXIT_FAILURE);
    }

    err->description = strdup(warning_msg);
    longjmp(err->jmpbuf, 1);
}

/**
 * `png_write()' writes a PNG image to the specified file.
 */
static int png_write(
    FILE * f,
    int width,
    int height,
    int bit_depth,
    int pixel_size,
    int color_type,
    double gamma,
    int bgcolor,
    char * title,
    char * author,
    char * description,
    char * copyright,
    char * email,
    char * url,
    unsigned char * buffer,
    unsigned char ** row_pointers,
    char ** error_msg)
{
    png_structp  png_ptr;
    png_infop  info_ptr;
    struct png_error error;

    if (error_msg)
        *error_msg = NULL;

    png_ptr = png_create_write_struct(
        PNG_LIBPNG_VER_STRING, &error,
        png_handle_error, png_handle_warning);
    if (!png_ptr)
        return -1;

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        return -1;
    }

    if (setjmp(error.jmpbuf)) {
        if (error_msg)
            *error_msg = error.description;
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return -1;
    }

    png_init_io(png_ptr, f);

    int interlace_type = PNG_INTERLACE_NONE;
    png_set_IHDR(
        png_ptr, info_ptr, width, height, bit_depth,
        color_type, interlace_type,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    if (gamma > 0.0)
        png_set_gAMA(png_ptr, info_ptr, gamma);

    if (bgcolor >= 0) {
        png_color_16 background;
        background.red = (bgcolor >> 16) & 0xFF;
        background.green = (bgcolor >> 8) & 0xFF;
        background.blue = bgcolor & 0xFF;
        png_set_bKGD(png_ptr, info_ptr, &background);
    }

    png_time t;
    png_convert_from_time_t(&t, time(NULL));
    png_set_tIME(png_ptr, info_ptr, &t);

    png_text text[6];
    int num_texts = 0;
    if (title) {
        text[num_texts].compression = PNG_TEXT_COMPRESSION_NONE;
        text[num_texts].key = "Title";
        text[num_texts].text = title;
        num_texts++;
    }
    if (author) {
        text[num_texts].compression = PNG_TEXT_COMPRESSION_NONE;
        text[num_texts].key = "Author";
        text[num_texts].text = author;
        num_texts++;
    }
    if (description) {
        text[num_texts].compression = PNG_TEXT_COMPRESSION_NONE;
        text[num_texts].key = "Description";
        text[num_texts].text = description;
        num_texts++;
    }
    if (copyright) {
        text[num_texts].compression = PNG_TEXT_COMPRESSION_NONE;
        text[num_texts].key = "Copyright";
        text[num_texts].text = copyright;
        num_texts++;
    }
    if (email) {
        text[num_texts].compression = PNG_TEXT_COMPRESSION_NONE;
        text[num_texts].key = "E-mail";
        text[num_texts].text = email;
        num_texts++;
    }
    if (url) {
        text[num_texts].compression = PNG_TEXT_COMPRESSION_NONE;
        text[num_texts].key = "URL";
        text[num_texts].text = url;
        num_texts++;
    }
    if (num_texts > 0)
        png_set_text(png_ptr, info_ptr, text, num_texts);

    png_write_info(png_ptr, info_ptr);
    png_set_packing(png_ptr);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
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

    /* 2. Read a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtxfile_read: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    struct mtxfile mtxfile;
    int lines_read;
    int64_t bytes_read;
    err = mtxfile_read(
        &mtxfile, args.precision, args.mtx_path, args.gzip,
        &lines_read, &bytes_read);
    if (err && lines_read >= 0) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s:%d: %s\n",
                program_invocation_short_name,
                args.mtx_path, lines_read+1,
                mtx_strerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    } else if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name,
                args.mtx_path, mtx_strerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }

    /* 3. Draw the matrix sparsity pattern. */
    if (args.verbose > 0) {
        fprintf(diagf, "mtx_matrix_spy: ");
        fflush(diagf);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int width;
    int height;
    int bit_depth;
    int pixel_size;
    int color_type;
    unsigned char * imgbuf;
    unsigned char ** row_pointers;
    err = draw_sparsity_pattern(
        &mtxfile, args.fgcolor, args.bgcolor,
        args.max_width, args.max_height,
        &width, &height, &bit_depth,
        &pixel_size, &color_type,
        &imgbuf, &row_pointers);
    if (err) {
        if (args.verbose > 0)
            fprintf(diagf, "\n");
        fprintf(stderr, "%s: %s",
                program_invocation_short_name, mtx_strerror(err));
        mtxfile_free(&mtxfile);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
    }

    mtxfile_free(&mtxfile);

    /* 3. Write the PNG image to file. */
    if (args.png_output_path) {
        if (args.verbose > 0) {
            fprintf(diagf, "png_write: ");
            fflush(diagf);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        FILE * f;
        if ((f = fopen(args.png_output_path, "w")) == NULL) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.png_output_path, strerror(errno));
            free(row_pointers);
            free(imgbuf);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        char * error_msg;
        err = png_write(
            f, width, height, bit_depth, pixel_size, color_type,
            args.gamma, args.bgcolor,
            args.title, args.author, args.description,
            args.copyright, args.email, args.url,
            imgbuf, row_pointers, &error_msg);
        if (err) {
            if (args.verbose > 0)
                fprintf(diagf, "\n");
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name,
                    args.png_output_path,
                    error_msg ? error_msg : strerror(errno));
            free(row_pointers);
            free(imgbuf);
            fclose(f);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        free(row_pointers);
        free(imgbuf);
        fclose(f);

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(diagf, "%'.6f seconds\n", timespec_duration(t0, t1));
            fflush(diagf);
        }
    } else {
        free(row_pointers);
        free(imgbuf);
    }

    /* 4. Clean up. */
    program_options_free(&args);
    return EXIT_SUCCESS;
}
