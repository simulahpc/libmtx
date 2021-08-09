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
 * Main libmtx header file.
 */

#ifndef LIBMTX_LIBMTX_H
#define LIBMTX_LIBMTX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/blas.h>
#include <libmtx/error.h>
#include <libmtx/header.h>
#include <libmtx/index_set.h>
#include <libmtx/io.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/array/array.h>
#include <libmtx/matrix/coordinate/coordinate.h>
#include <libmtx/mpi.h>
#include <libmtx/mtx.h>
#include <libmtx/reorder.h>
#include <libmtx/superlu_dist.h>
#include <libmtx/vector/vector.h>
#include <libmtx/vector/array/array.h>
#include <libmtx/vector/coordinate/coordinate.h>
#include <libmtx/version.h>

#endif
