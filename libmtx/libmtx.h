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
 * Last modified: 2021-08-19
 *
 * Main libmtx header file.
 */

#ifndef LIBMTX_LIBMTX_H
#define LIBMTX_LIBMTX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/array.h>
#include <libmtx/matrix/coordinate.h>
#include <libmtx/mtx/assembly.h>
#include <libmtx/mtx/blas.h>
#include <libmtx/mtx/cg.h>
#include <libmtx/mtx/header.h>
#include <libmtx/mtx/io.h>
#include <libmtx/mtx/mpi.h>
#include <libmtx/mtx/mtx.h>
#include <libmtx/mtx/precision.h>
#include <libmtx/mtx/reorder.h>
#include <libmtx/mtx/sort.h>
#include <libmtx/mtx/submatrix.h>
#include <libmtx/mtx/transpose.h>
#include <libmtx/mtx/triangle.h>
#include <libmtx/util/index_set.h>
#include <libmtx/vector/array.h>
#include <libmtx/vector/coordinate.h>
#include <libmtx/version.h>

#endif
