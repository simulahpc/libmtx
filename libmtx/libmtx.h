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
 * Last modified: 2022-05-20
 *
 * Main Libmtx header file.
 */

#ifndef LIBMTX_LIBMTX_H
#define LIBMTX_LIBMTX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/matrix/blas.h>
#include <libmtx/matrix/base/dense.h>
#include <libmtx/matrix/dist.h>
#include <libmtx/matrix/matrix.h>
#include <libmtx/matrix/base/coo.h>
#include <libmtx/matrix/matrix_csr.h>
#include <libmtx/matrix/symmetry.h>
#include <libmtx/matrix/transpose.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/solver/cg.h>
#include <libmtx/util/partition.h>
#include <libmtx/vector/base.h>
#include <libmtx/vector/blas.h>
#include <libmtx/vector/dist.h>
#include <libmtx/vector/field.h>
#include <libmtx/vector/omp.h>
#include <libmtx/vector/packed.h>
#include <libmtx/vector/precision.h>
#include <libmtx/vector/vector.h>
#include <libmtx/version.h>

#endif
