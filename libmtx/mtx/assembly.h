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
 * Assembling sparse matrices and vectors.
 */

#ifndef LIBMTX_MTX_ASSEMBLY_H
#define LIBMTX_MTX_ASSEMBLY_H

/**
 * `mtx_assembly' is used to enumerate assembly states for sparse
 * matrices in Matrix Market format.
 */
enum mtx_assembly
{
    mtx_unassembled,    /* unassembled; duplicate nonzeros allowed. */
    mtx_assembled,      /* assembled; duplicate nonzeros not allowed. */
};

/**
 * `mtx_assembly_str()' is a string representing the assembly type.
 */
const char * mtx_assembly_str(
    enum mtx_assembly assembly);

#endif
