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
 * Last modified: 2022-10-03
 *
 * Main Libmtx header file.
 */

#ifndef LIBMTX_LIBMTX_H
#define LIBMTX_LIBMTX_H

#include <libmtx/libmtx-config.h>

#include <libmtx/error.h>
#include <libmtx/linalg/base/coo.h>
#include <libmtx/linalg/base/csr.h>
#include <libmtx/linalg/base/dense.h>
#include <libmtx/linalg/base/vector.h>
#include <libmtx/linalg/blas/dense.h>
#include <libmtx/linalg/blas/vector.h>
#include <libmtx/linalg/field.h>
#include <libmtx/linalg/gemvoverlap.h>
#include <libmtx/linalg/local/matrix.h>
#include <libmtx/linalg/local/vector.h>
#include <libmtx/linalg/mpi/matrix.h>
#include <libmtx/linalg/mpi/vector.h>
#include <libmtx/linalg/null/coo.h>
#include <libmtx/linalg/null/vector.h>
#include <libmtx/linalg/omp/vector.h>
#include <libmtx/linalg/partition.h>
#include <libmtx/linalg/precision.h>
#include <libmtx/linalg/symmetry.h>
#include <libmtx/linalg/transpose.h>
#include <libmtx/mtxfile/mtxdistfile.h>
#include <libmtx/mtxfile/mtxfile.h>
#include <libmtx/solver/cg.h>
#include <libmtx/util/partition.h>
#include <libmtx/version.h>

#endif
