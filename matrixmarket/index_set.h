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
 * Index sets.
 */

#ifndef MATRIXMARKET_INDEX_SET_H
#define MATRIXMARKET_INDEX_SET_H

#include <stdbool.h>

/**
 * `mtx_index_set_type' enumerates different kinds of index sets.
 */
enum mtx_index_set_type
{
    mtx_index_set_interval,
};

/**
 * `mtx_index_set_type_str()' is a string representing the index set
 * type.
 */
const char * mtx_index_set_type_str(
    enum mtx_index_set_type index_set_type);

/**
 * `mtx_index_set_interval' represents an index set of contiguous
 * integers from a half-open interval [a,b).
 */
struct mtx_index_set_interval
{
    int a; /* left endpoint of the interval */
    int b; /* right endpoint of the interval */
};

/**
 * `mtx_index_set' is a data structure for index sets.
 */
struct mtx_index_set
{
    /**
     * `type' is the type of index set: `interval'.
     */
    enum mtx_index_set_type type;

    /**
     * `interval' is an index set of contiguous integers.
     */
    struct mtx_index_set_interval interval;
};

/**
 * `mtx_index_set_free()` frees resources associated with an index
 * set.
 */
void mtx_index_set_free(
    struct mtx_index_set * index_set);

/**
 * `mtx_index_set_init_interval()` creates an index set of contiguous
 * integers from an interval [a,b).
 */
int mtx_index_set_init_interval(
    struct mtx_index_set * index_set, int a, int b);

/**
 * `mtx_index_set_size()` returns the size of the index set.
 */
int mtx_index_set_size(
    const struct mtx_index_set * index_set,
    int * size);

/**
 * `mtx_index_set_contains()` returns `true' if the given integer is
 * contained in the index set and `false' otherwise.
 */
bool mtx_index_set_contains(
    const struct mtx_index_set * index_set, int n);

#endif
