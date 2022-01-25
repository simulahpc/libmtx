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
 * Last modified: 2022-01-25
 *
 * Permutations of finite sets.
 */

#include "libmtx/util/permute.h"

#include <libmtx/error.h>

#include <stdint.h>
#include <stdlib.h>

/**
 * ‘mtxpermutation_free()’ frees resources associated with a
 * permutation.
 */
void mtxpermutation_free(
    struct mtxpermutation * permutation)
{
    free(permutation->workspace);
    free(permutation->perm);
}

/**
 * ‘mtxpermutation_init()’ creates a permutation.
 *
 * The array ‘perm’ must be of length ‘size’ and must define a
 * permutation of the integers 0, 1, ..., ‘size-1’.
 *
 * Applying the permutation to an array ‘x’ of length ‘size’ moves the
 * element located at position ‘i’ to the position ‘perm[i]’.  In
 * other words, ‘x[perm[i]] <- x[i]’, for ‘i=0,1,...,size-1’.
 */
int mtxpermutation_init(
    struct mtxpermutation * permutation,
    int64_t size,
    const int64_t * perm)
{
    for (int64_t i = 0; i < size; i++) {
        if (perm[i] < 0 || perm[i] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    permutation->size = size;
    permutation->perm = malloc(size * sizeof(int64_t));
    if (!permutation->perm) return MTX_ERR_ERRNO;
    for (int64_t i = 0; i < size; i++)
        permutation->perm[i] = perm[i];
    permutation->workspace_size = 0;
    permutation->workspace = NULL;
    return MTX_SUCCESS;
}

static int mtxpermutation_alloc_workspace(
    struct mtxpermutation * permutation,
    int64_t size)
{
    if (permutation->workspace && permutation->workspace_size >= size)
        return MTX_SUCCESS;
    else if (permutation->workspace)
        free(permutation->workspace);
    permutation->workspace = malloc(size);
    if (!permutation->workspace)
        return MTX_ERR_ERRNO;
    permutation->workspace_size = size;
    return MTX_SUCCESS;
}

/**
 * ‘mtxpermutation_invert()’ inverts a permutation.
 */
int mtxpermutation_invert(
    struct mtxpermutation * permutation)
{
    int err = mtxpermutation_alloc_workspace(
        permutation, permutation->size * sizeof(int64_t));
    if (err) return err;
    int64_t * perm = permutation->perm;
    int64_t * invperm = (int64_t *) permutation->workspace;
    for (int64_t i = 0; i < permutation->size; i++)
        invperm[perm[i]] = i;
    for (int64_t i = 0; i < permutation->size; i++)
        perm[i] = invperm[i];
    return MTX_SUCCESS;
}

/**
 * ‘mtxpermutation_compose()’ composes two permutations to obtain the
 * product or combined permutation. If ‘a’ and ‘b’ are permutations of
 * the 0, 1, ..., N-1, then their product or composition ‘c’ is
 * defined as ‘c[i] = b[a[i]]’, for ‘i=0,1,...,N-1’.
 */
int mtxpermutation_compose(
    struct mtxpermutation * c,
    struct mtxpermutation * a,
    struct mtxpermutation * b)
{
    int err;
    if (a->size != b->size)
        return MTX_ERR_INCOMPATIBLE_SIZE;
    int64_t size = a->size;
    int64_t * cperm;
    if (a->workspace && a->workspace_size >= size * sizeof(int64_t)) {
        cperm = (int64_t *) a->workspace;
    } else if (b->workspace && b->workspace_size >= size * sizeof(int64_t)) {
        cperm = (int64_t *) b->workspace;
    } else {
        err = mtxpermutation_alloc_workspace(a, size * sizeof(int64_t));
        if (err) return err;
        cperm = (int64_t *) a->workspace;
    }
    for (int64_t i = 0; i < size; i++)
        cperm[i] = b->perm[a->perm[i]];
    return mtxpermutation_init(c, size, cperm);
}

/**
 * ‘mtxpermutation_permute_int()’ permutes an array of integers.
 *
 * The array ‘x’ must be of length ‘size’, which may not exceed
 * ‘permutation->size’. Applying the permutation to ‘x’ moves the
 * element at position ‘i’ to the position ‘permutation->perm[i]’, or,
 * ‘x[permutation->perm[i]] <- x[i]’, for ‘i=0,1,...,size-1’.
 */
int mtxpermutation_permute_int(
    struct mtxpermutation * permutation,
    int64_t size,
    int * x)
{
    int err = mtxpermutation_alloc_workspace(permutation, size * sizeof(*x));
    if (err) return err;
    const int64_t * perm = permutation->perm;
    int * y = (int *) permutation->workspace;
    for (int64_t i = 0; i < size; i++)
        y[i] = x[i];
    for (int64_t i = 0; i < size; i++)
        x[perm[i]] = y[i];
    return MTX_SUCCESS;
}

/*
 * Standalone permutation functions
 */

/**
 * ‘permute_int()’ permutes an array of integers.
 *
 * The arrays ‘perm’ and ‘x’ must be of length ‘size’. The former
 * defines a permutation of the integers 0, 1, ..., ‘size-1’, whereas
 * the latter is an array of integers to which the permutation is
 * applied. The result of applying the permutation is that the element
 * that before was located at position ‘i’ is now located at the
 * position ‘perm[i]’, or, in other words, ‘x[perm[i]] <- x[i]’, for
 * ‘i=0,1,...,size-1’.
 *
 * Note that this function allocates temporary storage of the same
 * size as the array ‘x’ to use for carrying out the permutation in
 * linear time. If a permutation is applied repeatedly, then it is
 * instead recommended to use ‘struct mtxpermutation’ to avoid
 * overhead associated with error checking and allocating storage
 * every time.
 */
int permute_int(
    int64_t size,
    int64_t * perm,
    int * x)
{
    for (int64_t i = 0; i < size; i++) {
        if (perm[i] < 0 || perm[i] >= size)
            return MTX_ERR_INDEX_OUT_OF_BOUNDS;
    }
    int * y = malloc(size * sizeof(int));
    if (!y) return MTX_ERR_ERRNO;
    for (int64_t i = 0; i < size; i++)
        y[i] = x[i];
    for (int64_t i = 0; i < size; i++)
        x[perm[i]] = y[i];
    free(y);
    return MTX_SUCCESS;
}
