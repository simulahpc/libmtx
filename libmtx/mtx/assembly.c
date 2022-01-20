/* This file is part of Libmtx.
 *
 * Copyright (C) 2021 James D. Trotter
 *
 * Libmtx is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Libmtx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Libmtx.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2021-08-09
 *
 * Assembling sparse matrices and vectors.
 */

#include <libmtx/mtx/assembly.h>

/**
 * `mtx_assembly_str()' is a string representing the assembly type.
 */
const char * mtx_assembly_str(
    enum mtx_assembly assembly)
{
    switch (assembly) {
    case mtx_unassembled: return "unassembled";
    case mtx_assembled: return "assembled";
    default: return "unknown";
    }
}
