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
 * Last modified: 2022-01-16
 *
 * Printing formatted output.
 */

#include "fmtspec.h"
#include "parse.h"

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * `fmtspec_init()' creates a format specifier.
 */
struct fmtspec fmtspec_init(
    enum fmtspec_flags flags,
    enum fmtspec_width width,
    enum fmtspec_precision precision,
    enum fmtspec_length length,
    enum fmtspec_type specifier)
{
    struct fmtspec f = {
        .flags = flags,
        .width = width,
        .precision = precision,
        .length = length,
        .specifier = specifier};
    return f;
}

static const char * length_str(
    enum fmtspec_length length)
{
    switch (length) {
    case fmtspec_length_none: return "";
    case fmtspec_length_hh: return "hh";
    case fmtspec_length_h: return "h";
    case fmtspec_length_l: return "l";
    case fmtspec_length_ll: return "ll";
    case fmtspec_length_j: return "j";
    case fmtspec_length_z: return "z";
    case fmtspec_length_t: return "t";
    case fmtspec_length_L: return "L";
    default: return strerror(EINVAL);
    }
}

static const char * specifier_str(
    enum fmtspec_type specifier)
{
    switch (specifier) {
    case fmtspec_d: return "d";
    case fmtspec_u: return "u";
    case fmtspec_o: return "o";
    case fmtspec_x: return "x";
    case fmtspec_X: return "X";
    case fmtspec_f: return "f";
    case fmtspec_F: return "F";
    case fmtspec_e: return "e";
    case fmtspec_E: return "E";
    case fmtspec_g: return "g";
    case fmtspec_G: return "G";
    case fmtspec_a: return "a";
    case fmtspec_A: return "A";
    case fmtspec_c: return "c";
    case fmtspec_s: return "s";
    case fmtspec_p: return "p";
    case fmtspec_n: return "n";
    case fmtspec_percent: return "%";
    default: return strerror(EINVAL);
    }
}

/**
 * `fmtspec_str()` is a string representing the given format
 * specifier.
 */
char * fmtspec_str(
    struct fmtspec format)
{
    if ((format.width == fmtspec_width_none ||
         format.width == fmtspec_width_star) &&
        (format.precision == fmtspec_precision_none ||
         format.precision == fmtspec_precision_star))
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%s%s%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if (format.width >= 0 &&
               (format.precision == fmtspec_precision_none ||
                format.precision == fmtspec_precision_star))
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%d%s%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if ((format.width == fmtspec_width_none ||
                format.width == fmtspec_width_star) &&
               format.precision >= 0)
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%s.%d%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
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
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
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

static int parse_fmtspec_flags(
    const char ** s,
    enum fmtspec_flags * flags)
{
    *flags = 0;
    while (**s == '-' || **s == '+' || **s == ' ' || **s == '#' || **s == '0')
    {
        switch (**s) {
        case '-': *flags |= fmtspec_flags_minus; break;
        case '+': *flags |= fmtspec_flags_plus; break;
        case ' ': *flags |= fmtspec_flags_space; break;
        case '#': *flags |= fmtspec_flags_number_sign; break;
        case '0': *flags |= fmtspec_flags_zero; break;
        default: return EINVAL;
        }
        (*s)++;
    }
    return 0;
}

static int parse_fmtspec_width(
    const char ** s,
    enum fmtspec_width * width)
{
    if (**s == '*') {
        *width = fmtspec_width_star;
        (*s)++;
        return 0;
    }

    int32_t n;
    const char * t;
    int err = parse_int32_ex(*s, NULL, &n, &t);
    if (err) {
        *width = fmtspec_width_none;
        return 0;
    }
    if (n < 0)
        return EINVAL;
    *width = n;
    *s = t;
    return 0;
}

static int parse_fmtspec_precision(
    const char ** s,
    enum fmtspec_precision * precision)
{
    if (**s != '.') {
        *precision = fmtspec_precision_none;
        return 0;
    }
    (*s)++;

    if (**s == '*') {
        *precision = fmtspec_precision_star;
        (*s)++;
        return 0;
    }

    int32_t n;
    const char * t;
    int err = parse_int32_ex(*s, NULL, &n, &t);
    if (err) {
        *precision = fmtspec_precision_none;
        return 0;
    }
    if (n < 0)
        return EINVAL;
    *precision = n;
    *s = t;
    return 0;
}

static int parse_fmtspec_length(
    const char ** s,
    enum fmtspec_length * length)
{
    if (**s == 'h' && *((*s)+1) == 'h') {
        *length = fmtspec_length_hh;
        *s += 2;
    } else if (**s == 'h' && *((*s)+1) != 'h') {
        *length = fmtspec_length_h;
        (*s)++;
    } else if (**s == 'l' && *((*s)+1) != 'l') {
        *length = fmtspec_length_l;
        (*s)++;
    } else if (**s == 'l' && *((*s)+1) == 'l') {
        *length = fmtspec_length_ll;
        *s += 2;
    } else if (**s == 'j') {
        *length = fmtspec_length_j;
        (*s)++;
    } else if (**s == 'z') {
        *length = fmtspec_length_z;
        (*s)++;
    } else if (**s == 't') {
        *length = fmtspec_length_t;
        (*s)++;
    } else if (**s == 'L') {
        *length = fmtspec_length_L;
        (*s)++;
    } else {
        *length = fmtspec_length_none;
    }
    return 0;
}

static int parse_fmtspec_type(
    const char ** s,
    enum fmtspec_type * specifier)
{
    switch (**s) {
    case 'd': *specifier = fmtspec_d; break;
    case 'i': *specifier = fmtspec_i; break;
    case 'u': *specifier = fmtspec_u; break;
    case 'o': *specifier = fmtspec_o; break;
    case 'x': *specifier = fmtspec_x; break;
    case 'X': *specifier = fmtspec_X; break;
    case 'f': *specifier = fmtspec_f; break;
    case 'F': *specifier = fmtspec_F; break;
    case 'e': *specifier = fmtspec_e; break;
    case 'E': *specifier = fmtspec_E; break;
    case 'g': *specifier = fmtspec_g; break;
    case 'G': *specifier = fmtspec_G; break;
    case 'a': *specifier = fmtspec_a; break;
    case 'A': *specifier = fmtspec_A; break;
    case 'c': *specifier = fmtspec_c; break;
    case 's': *specifier = fmtspec_s; break;
    case 'p': *specifier = fmtspec_p; break;
    case 'n': *specifier = fmtspec_n; break;
    case '%': *specifier = fmtspec_percent; break;
    default: return EINVAL;
    }
    (*s)++;
    return 0;
}

/**
 * `parse_fmtspec()' parses a string containing a format
 * specifier.
 *
 * If `endptr` is not `NULL`, the address stored in `endptr` points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, `parse_fmtspec()' returns `0'.  Otherwise, if
 * the input contained invalid characters, `EINVAL' is returned.
 */
int parse_fmtspec(
    const char * s,
    struct fmtspec * format,
    const char ** endptr)
{
    int err;
    const char * t = s;
    if (*t != '%')
        return EINVAL;
    t++;

    err = parse_fmtspec_flags(&t, &format->flags);
    if (err)
        return err;
    err = parse_fmtspec_width(&t, &format->width);
    if (err)
        return err;
    err = parse_fmtspec_precision(&t, &format->precision);
    if (err)
        return err;
    err = parse_fmtspec_length(&t, &format->length);
    if (err)
        return err;
    err = parse_fmtspec_type(&t, &format->specifier);
    if (err)
        return err;
    if (endptr)
        *endptr = t;
    return 0;
}
