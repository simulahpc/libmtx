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
 * Printing formatted output.
 */

#include "format.h"
#include "../parse.h"

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `format_specifier_init()' creates a format specifier.
 */
struct format_specifier format_specifier_init(
    enum format_specifier_flags flags,
    enum format_specifier_width width,
    enum format_specifier_precision precision,
    enum format_specifier_length length,
    enum format_specifier_type specifier)
{
    struct format_specifier f = {
        .flags = flags,
        .width = width,
        .precision = precision,
        .length = length,
        .specifier = specifier};
    return f;
}

static const char * length_str(
    enum format_specifier_length length)
{
    switch (length) {
    case format_specifier_length_none: return "";
    case format_specifier_length_hh: return "hh";
    case format_specifier_length_h: return "h";
    case format_specifier_length_l: return "l";
    case format_specifier_length_ll: return "ll";
    case format_specifier_length_j: return "j";
    case format_specifier_length_z: return "z";
    case format_specifier_length_t: return "t";
    case format_specifier_length_L: return "L";
    default: return strerror(EINVAL);
    }
}

static const char * specifier_str(
    enum format_specifier_type specifier)
{
    switch (specifier) {
    case format_specifier_d: return "d";
    case format_specifier_u: return "u";
    case format_specifier_o: return "o";
    case format_specifier_x: return "x";
    case format_specifier_X: return "X";
    case format_specifier_f: return "f";
    case format_specifier_F: return "F";
    case format_specifier_e: return "e";
    case format_specifier_E: return "E";
    case format_specifier_g: return "g";
    case format_specifier_G: return "G";
    case format_specifier_a: return "a";
    case format_specifier_A: return "A";
    case format_specifier_c: return "c";
    case format_specifier_s: return "s";
    case format_specifier_p: return "p";
    case format_specifier_n: return "n";
    case format_specifier_percent: return "%";
    default: return strerror(EINVAL);
    }
}

/**
 * `format_specifier_str()` is a string representing the given format
 * specifier.
 */
char * format_specifier_str(
    struct format_specifier format)
{
    if ((format.width == format_specifier_width_none ||
         format.width == format_specifier_width_star) &&
        (format.precision == format_specifier_precision_none ||
         format.precision == format_specifier_precision_star))
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%s%s%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width == format_specifier_width_none ? "" : "*",
            format.precision == format_specifier_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width == format_specifier_width_none ? "" : "*",
            format.precision == format_specifier_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if (format.width >= 0 &&
               (format.precision == format_specifier_precision_none ||
                format.precision == format_specifier_precision_star))
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%d%s%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width,
            format.precision == format_specifier_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width,
            format.precision == format_specifier_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if ((format.width == format_specifier_width_none ||
                format.width == format_specifier_width_star) &&
               format.precision >= 0)
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%s.%d%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width == format_specifier_width_none ? "" : "*",
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width == format_specifier_width_none ? "" : "*",
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if (format.width >= 0 && format.precision >= 0) {
        const char * format_str = "%%" "%s%s%s%s%s" "%d.%d%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width,
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & format_specifier_flags_minus ? "-" : "",
            format.flags & format_specifier_flags_plus ? "+" : "",
            format.flags & format_specifier_flags_space ? " " : "",
            format.flags & format_specifier_flags_number_sign ? "#" : "",
            format.flags & format_specifier_flags_zero ? "0" : "",
            format.width,
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else {
        size_t len = 1024;
        char * s = malloc(len+1);
        strncpy(s, strerror(EINVAL), len);
        return s;
    }
}

static int parse_format_specifier_flags(
    const char ** s,
    enum format_specifier_flags * flags)
{
    *flags = 0;
    while (**s == '-' || **s == '+' || **s == ' ' || **s == '#' || **s == '0')
    {
        switch (**s) {
        case '-': *flags |= format_specifier_flags_minus; break;
        case '+': *flags |= format_specifier_flags_plus; break;
        case ' ': *flags |= format_specifier_flags_space; break;
        case '#': *flags |= format_specifier_flags_number_sign; break;
        case '0': *flags |= format_specifier_flags_zero; break;
        default: return EINVAL;
        }
        (*s)++;
    }
    return 0;
}

static int parse_format_specifier_width(
    const char ** s,
    enum format_specifier_width * width)
{
    if (**s == '*') {
        *width = format_specifier_width_star;
        (*s)++;
        return 0;
    }

    int32_t n;
    const char * t;
    int err = parse_int32(*s, NULL, &n, &t);
    if (err) {
        *width = format_specifier_width_none;
        return 0;
    }
    if (n < 0)
        return EINVAL;
    *width = n;
    *s = t;
    return 0;
}

static int parse_format_specifier_precision(
    const char ** s,
    enum format_specifier_precision * precision)
{
    if (**s != '.') {
        *precision = format_specifier_precision_none;
        return 0;
    }
    (*s)++;

    if (**s == '*') {
        *precision = format_specifier_precision_star;
        (*s)++;
        return 0;
    }

    int32_t n;
    const char * t;
    int err = parse_int32(*s, NULL, &n, &t);
    if (err) {
        *precision = format_specifier_precision_none;
        return 0;
    }
    if (n < 0)
        return EINVAL;
    *precision = n;
    *s = t;
    return 0;
}

static int parse_format_specifier_length(
    const char ** s,
    enum format_specifier_length * length)
{
    if (**s == 'h' && *((*s)+1) == 'h') {
        *length = format_specifier_length_hh;
        *s += 2;
    } else if (**s == 'h' && *((*s)+1) != 'h') {
        *length = format_specifier_length_h;
        (*s)++;
    } else if (**s == 'l' && *((*s)+1) != 'l') {
        *length = format_specifier_length_l;
        (*s)++;
    } else if (**s == 'l' && *((*s)+1) == 'l') {
        *length = format_specifier_length_ll;
        *s += 2;
    } else if (**s == 'j') {
        *length = format_specifier_length_j;
        (*s)++;
    } else if (**s == 'z') {
        *length = format_specifier_length_z;
        (*s)++;
    } else if (**s == 't') {
        *length = format_specifier_length_t;
        (*s)++;
    } else if (**s == 'L') {
        *length = format_specifier_length_L;
        (*s)++;
    } else {
        *length = format_specifier_length_none;
    }
    return 0;
}

static int parse_format_specifier_type(
    const char ** s,
    enum format_specifier_type * specifier)
{
    switch (**s) {
    case 'd': *specifier = format_specifier_d; break;
    case 'i': *specifier = format_specifier_i; break;
    case 'u': *specifier = format_specifier_u; break;
    case 'o': *specifier = format_specifier_o; break;
    case 'x': *specifier = format_specifier_x; break;
    case 'X': *specifier = format_specifier_X; break;
    case 'f': *specifier = format_specifier_f; break;
    case 'F': *specifier = format_specifier_F; break;
    case 'e': *specifier = format_specifier_e; break;
    case 'E': *specifier = format_specifier_E; break;
    case 'g': *specifier = format_specifier_g; break;
    case 'G': *specifier = format_specifier_G; break;
    case 'a': *specifier = format_specifier_a; break;
    case 'A': *specifier = format_specifier_A; break;
    case 'c': *specifier = format_specifier_c; break;
    case 's': *specifier = format_specifier_s; break;
    case 'p': *specifier = format_specifier_p; break;
    case 'n': *specifier = format_specifier_n; break;
    case '%': *specifier = format_specifier_percent; break;
    default: return EINVAL;
    }
    (*s)++;
    return 0;
}

/**
 * `parse_format_specifier()' parses a string containing a format
 * specifier.
 *
 * If `endptr` is not `NULL`, the address stored in `endptr` points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, `parse_format_specifier()' returns `0'.  Otherwise, if
 * the input contained invalid characters, `EINVAL' is returned.
 */
int parse_format_specifier(
    const char * s,
    struct format_specifier * format,
    const char ** endptr)
{
    int err;
    const char * t = s;
    if (*t != '%')
        return EINVAL;
    t++;

    err = parse_format_specifier_flags(&t, &format->flags);
    if (err)
        return err;
    err = parse_format_specifier_width(&t, &format->width);
    if (err)
        return err;
    err = parse_format_specifier_precision(&t, &format->precision);
    if (err)
        return err;
    err = parse_format_specifier_length(&t, &format->length);
    if (err)
        return err;
    err = parse_format_specifier_type(&t, &format->specifier);
    if (err)
        return err;
    if (endptr)
        *endptr = t;
    return 0;
}
