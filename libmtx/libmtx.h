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
 * Last modified: 2022-01-20
 *
 * Main Libmtx header file.
 */

#ifndef LIBMTX_LIBMTX_H
#define LIBMTX_LIBMTX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/field.h>
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/blas.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/matrix/dist.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/matrix_coordinate.h>
#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mpi.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/submatrix.h>
#include <libmtx/mtx/transpose.h>
#include <libmtx/mtx/triangle.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxdistfile2.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/precision.h>
#include <libmtx/solver/cg.h>
#include <libmtx/util/index_set.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/coordinate.h>
#include <libmtx/vector/dist.h>
#include <libmtx/vector/distvector.h>
#include <libmtx/vector/vector.h>
#include <libmtx/vector/vector_array.h>
#include <libmtx/vector/vector_coordinate.h>
#include <libmtx/version.h>

#endif
