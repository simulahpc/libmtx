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
 * Last modified: 2021-08-02
 *
 * Index sets.
 */

#include <matrixmarket/error.h>
#include <matrixmarket/index_set.h>

#include <errno.h>

#include <stdbool.h>

/**
 * `mtx_index_set_type_str()' is a string representing the index set
 * type.
 */
const char * mtx_index_set_type_str(
    enum mtx_index_set_type index_set_type)
{
    switch (index_set_type) {
    case mtx_index_set_interval: return "interval";
    default: return "unknown";
    }
}

/**
 * `mtx_index_set_free()` frees resources associated with an index set.
 */
void mtx_index_set_free(
    struct mtx_index_set * index_set)
{
}

/**
 * `mtx_index_set_init_interval()` creates an index set of contiguous
 * integers from an interval [a,b).
 */
int mtx_index_set_init_interval(
    struct mtx_index_set * index_set, int a, int b)
{
    index_set->type = mtx_index_set_interval;
    index_set->interval.a = a;
    index_set->interval.b = b;
    return MTX_SUCCESS;
}

/**
 * `mtx_index_set_size()` returns the size of the index set.
 */
int mtx_index_set_size(
    const struct mtx_index_set * index_set,
    int * size)
{
    if (index_set->type == mtx_index_set_interval) {
        const struct mtx_index_set_interval * interval = &index_set->interval;
        *size = interval->b - interval->a;
    } else {
        return MTX_ERR_INVALID_INDEX_SET_TYPE;
    }
    return MTX_SUCCESS;
}

/**
 * `mtx_index_set_contains()` returns `true' if the given integer is
 * contained in the index set and `false' otherwise.
 */
bool mtx_index_set_contains(
    const struct mtx_index_set * index_set, int n)
{
    if (index_set->type == mtx_index_set_interval) {
        const struct mtx_index_set_interval * interval = &index_set->interval;
        return (n >= interval->a) && (n < interval->b);
    } else {
        return false;
    }
    return false;
}
