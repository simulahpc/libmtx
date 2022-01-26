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

#ifndef LIBMTX_UTIL_PERMUTE_H
#define LIBMTX_UTIL_PERMUTE_H

#include <stdint.h>

struct mtxdisterror;

/**
 * ‘mtxpermutation’ is a data structure representing a permutation of
 * a finite set.
 */
struct mtxpermutation
{
    /**
     * ‘size’ is the number of elements in the underlying set.
     * Elements of the set are labelled 0, 1, ..., ‘size-1’.
     */
    int64_t size;

    /**
     * ‘perm’ is an array of length ‘size’ containing a permutation of
     * the integers 0, 1, ..., ‘size-1’. That is, each integer should
     * appear exactly once in ‘perm’.
     */
    int64_t * perm;

    /**
     * ‘workspace_size’ is the size (in bytes) of the additional,
     * temporary storage that has been allocated for the permutation.
     */
    int64_t workspace_size;

    /**
     * ‘workspace’ is a pointer to additional, temporary storage that
     * is used when permuting an array.
     */
    void * workspace;
};

/**
 * ‘mtxpermutation_free()’ frees resources associated with a
 * permutation.
 */
void mtxpermutation_free(
    struct mtxpermutation * permutation);

/**
 * ‘mtxpermutation_init_default()’ creates a default, identity
 * permutation that maps every element to itself.
 */
int mtxpermutation_init_default(
    struct mtxpermutation * permutation,
    int64_t size);

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
    const int64_t * perm);

/**
 * ‘mtxpermutation_invert()’ inverts a permutation.
 */
int mtxpermutation_invert(
    struct mtxpermutation * permutation);

/**
 * ‘mtxpermutation_compose()’ composes two permutations to obtain the
 * product or combined permutation. If ‘a’ and ‘b’ are permutations of
 * the 0, 1, ..., N-1, then their product or composition ‘c’ is
 * defined as ‘c[i] = b[a[i]]’, for ‘i=0,1,...,N-1’.
 */
int mtxpermutation_compose(
    struct mtxpermutation * c,
    struct mtxpermutation * a,
    struct mtxpermutation * b);

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
    int * x);

/**
 * ‘mtxpermutation_permute_int64()’ permutes an array of 64-bit signed
 * integers.
 *
 * The array ‘x’ must be of length ‘size’, which may not exceed
 * ‘permutation->size’. Applying the permutation to ‘x’ moves the
 * element at position ‘i’ to the position ‘permutation->perm[i]’, or,
 * ‘x[permutation->perm[i]] <- x[i]’, for ‘i=0,1,...,size-1’.
 */
int mtxpermutation_permute_int64(
    struct mtxpermutation * permutation,
    int64_t size,
    int64_t * x);

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
    int * x);

#endif
