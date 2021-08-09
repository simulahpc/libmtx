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

#ifndef LIBMTX_PRINT_H
#define LIBMTX_PRINT_H

#include <stdio.h>

/**
 * `format_specifier_flags' enumerates the flags that may be used to
 * modify format specifiers.
 */
enum format_specifier_flags
{
    /* Left-justify within the given field width; Right justification
     * is the default (see width sub-specifier). */
    format_specifier_flags_minus = 1 << 0,

    /* Forces to preceed the result with a plus or minus sign (+ or -)
     * even for positive numbers. By default, only negative numbers
     * are preceded with a - sign. */
    format_specifier_flags_plus = 1 << 1,

    /* If no sign is going to be written, a blank space is inserted
     * before the value. */
    format_specifier_flags_space = 1 << 2,

    /* 
     * Used with o, x or X specifiers the value is preceeded with 0,
     * 0x or 0X respectively for values different than zero.
     *
     * Used with a, A, e, E, f, F, g or G it forces the written output
     * to contain a decimal point even if no more digits follow. By
     * default, if no digits follow, no decimal point is written.
     */
    format_specifier_flags_number_sign = 1 << 3,

    /* Left-pads the number with zeroes (0) instead of spaces when
     * padding is specified (see width sub-specifier). */
    format_specifier_flags_zero = 1 << 4,
};

/**
 * `format_specifier_width' enumerates the types of field widths that
 *  may be used to modify format specifiers.
 *
 * The special enum values `format_specifier_width_none' and
 * `format_specifier_width_star' are assigned negative integer values,
 * whereas a non-negative field width is interpreted as described
 * below.
 *
 * A non-negative field width specifies the minimum number of
 * characters to be printed. If the value to be printed is shorter
 * than this number, the result is padded with blank spaces. The value
 * is not truncated even if the result is larger.
 */
enum format_specifier_width
{
    /* No width specifier. */
    format_specifier_width_none = -1,

    /* The width is not specified in the format string, but as an
     * additional integer value argument preceding the argument that
     * has to be formatted. */
    format_specifier_width_star = -2,
};

/**
 * `format_specifier_precision' enumerates the precision that may be
 *  used to modify format specifiers.
 *
 * The special enum values `format_specifier_precision_none' and
 * `format_specifier_precision_star' are assigned negative integer
 * values, whereas a non-negative precision is interpreted as
 * described below.
 *
 * For integer specifiers (d, i, o, u, x, X): precision specifies the
 * minimum number of digits to be written. If the value to be written
 * is shorter than this number, the result is padded with leading
 * zeros. The value is not truncated even if the result is longer. A
 * precision of 0 means that no character is written for the value 0.
 *
 * For a, A, e, E, f and F specifiers: this is the number of digits to
 * be printed after the decimal point (by default, this is 6).
 *
 * For g and G specifiers: This is the maximum number of significant
 * digits to be printed.
 *
 * For s: this is the maximum number of characters to be printed. By
 * default all characters are printed until the ending null character
 * is encountered.
 *
 * If the period is specified without an explicit value for precision,
 * 0 is assumed.
 */
enum format_specifier_precision
{
    /* No precision specifier. */
    format_specifier_precision_none = -1,

    /* The precision is not specified in the format string, but as an
     * additional integer value argument preceding the argument that
     * has to be formatted. */
    format_specifier_precision_star = -2,
};

/**
 * `format_specifier_length' enumerates lengths that may be used to
 *  modify format specifiers. The combination of length and specifier
 *  determine which data type is used when interpreting the values to
 *  be printed.
 */
enum format_specifier_length
{
    format_specifier_length_none,
    format_specifier_length_hh,
    format_specifier_length_h,
    format_specifier_length_l,
    format_specifier_length_ll,
    format_specifier_length_j,
    format_specifier_length_z,
    format_specifier_length_t,
    format_specifier_length_L,
};

/**
 * `format_specifier_type' enumerates the format specifiers that may
 *  be used to print formatted output with printf.
 */
enum format_specifier_type
{
    format_specifier_d = 0,   /* Signed decimal integer */
    format_specifier_i = 0,   /* Signed decimal integer */
    format_specifier_u,       /* Unsigned decimal integer */
    format_specifier_o,       /* Unsigned octal */
    format_specifier_x,       /* Unsigned hexadecimal integer, lowercase */
    format_specifier_X,       /* Unsigned hexadecimal integer, uppercase */
    format_specifier_f,       /* Decimal floating point, lowercase */
    format_specifier_F,       /* Decimal floating point, uppercase */
    format_specifier_e,       /* Scientific notation, lowercase */
    format_specifier_E,       /* Scientific notation, uppercase */
    format_specifier_g,       /* Use the shortest of %e or %f */
    format_specifier_G,       /* Use the shortest of %E or %F */
    format_specifier_a,       /* Hexadecimal floating point, lowercase */
    format_specifier_A,       /* Hexadecimal floating point, uppercase */
    format_specifier_c,       /* Character */
    format_specifier_s,       /* String of characters */
    format_specifier_p,       /* Pointer address */
    format_specifier_n,       /* Nothing printed. */
    format_specifier_percent, /* Write a single % character */
};

/**
 * `format_specifier' represents a format specifier that may be used
 * to print formatted output with printf.
 */
struct format_specifier
{
    enum format_specifier_flags flags;
    enum format_specifier_width width;
    enum format_specifier_precision precision;
    enum format_specifier_length length;
    enum format_specifier_type specifier;
};

/**
 * `format_specifier_init()' creates a format specifier.
 */
struct format_specifier format_specifier_init(
    enum format_specifier_flags flags,
    enum format_specifier_width width,
    enum format_specifier_precision precision,
    enum format_specifier_length length,
    enum format_specifier_type specifier);

/**
 * `format_specifier_str()` is a string representing the given format
 * specifier.
 *
 * Storage for the returned string is allocated with
 * `malloc()'. Therefore, the caller is responsible for calling
 * `free()' to deallocate the underlying storage.
 */
char * format_specifier_str(
    struct format_specifier format);

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
    const char ** endptr);

#endif
