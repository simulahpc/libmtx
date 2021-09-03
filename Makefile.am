# This file is part of libmtx.
#
# Copyright (C) 2021 James D. Trotter
#
# libmtx is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# libmtx is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libmtx.  If not, see
# <https://www.gnu.org/licenses/>.
#
# Authors: James D. Trotter <james@simula.no>
# Last modified: 2021-08-16
#
# Makefile for the top-level directory of libmtx.
#
# Process this file with automake to produce Makefile.in.

ACLOCAL_AMFLAGS = -I m4

# Additional files to distribute
EXTRA_DIST = \
	AUTHORS \
	COPYING \
	ChangeLog \
	INSTALL \
	README \
	autogen.sh \
	manual/texinfo.css \
	tests/test_mtx_mpi_parallel

# Matrix Market library
lib_LTLIBRARIES = libmtx.la
libmtx_la_CPPFLAGS = -I$(top_srcdir) $(MPICPPFLAGS)
libmtx_la_LDFLAGS = -version-info $(LIBMTX_LT_VERSION) $(MPILDFAGS)
libmtx_la_LIBADD = $(BLAS_LIBS) $(MPILDFLAGS)

# Source files
libmtx_la_SOURCES = \
	libmtx/error.c \
	libmtx/matrix/array.c \
	libmtx/matrix/array/blas.c \
	libmtx/matrix/array/data.c \
	libmtx/matrix/array/mpi.c \
	libmtx/matrix/array/reorder.c \
	libmtx/matrix/array/sort.c \
	libmtx/matrix/coordinate.c \
	libmtx/matrix/coordinate/blas.c \
	libmtx/matrix/coordinate/data.c \
	libmtx/matrix/coordinate/io.c \
	libmtx/matrix/coordinate/mpi.c \
	libmtx/matrix/coordinate/reorder.c \
	libmtx/matrix/coordinate/sort.c \
	libmtx/mtx/assembly.c \
	libmtx/mtx/blas.c \
	libmtx/mtx/cg.c \
	libmtx/mtx/header.c \
	libmtx/mtx/io.c \
	libmtx/mtx/mpi.c \
	libmtx/mtx/mtx.c \
	libmtx/mtx/precision.c \
	libmtx/mtx/reorder.c \
	libmtx/mtx/sort.c \
	libmtx/mtx/submatrix.c \
	libmtx/mtx/transpose.c \
	libmtx/mtx/triangle.c \
	libmtx/mtxfile/header.c \
	libmtx/mtxfile/comments.c \
	libmtx/mtxfile/mtxfile.c \
	libmtx/mtxfile/size.c \
	libmtx/mtxfile/data.c \
	libmtx/util/cuthill_mckee.c \
	libmtx/util/cuthill_mckee.h \
	libmtx/util/format.c \
	libmtx/util/format.h \
	libmtx/util/index_set.c \
	libmtx/util/io.c \
	libmtx/util/io.h \
	libmtx/util/parse.c \
	libmtx/util/parse.h \
	libmtx/vector/array.c \
	libmtx/vector/array/blas.c \
	libmtx/vector/array/data.c \
	libmtx/vector/array/reorder.c \
	libmtx/vector/coordinate.c \
	libmtx/vector/coordinate/blas.c \
	libmtx/vector/coordinate/data.c \
	libmtx/vector/coordinate/io.c \
	libmtx/vector/coordinate/reorder.c \
	libmtx/version.c

# Header files
nobase_include_HEADERS = \
	libmtx/error.h \
	libmtx/libmtx.h \
	libmtx/matrix/array.h \
	libmtx/matrix/array/blas.h \
	libmtx/matrix/array/data.h \
	libmtx/matrix/array/mpi.h \
	libmtx/matrix/array/reorder.h \
	libmtx/matrix/array/sort.h \
	libmtx/matrix/coordinate.h \
	libmtx/matrix/coordinate/blas.h \
	libmtx/matrix/coordinate/data.h \
	libmtx/matrix/coordinate/io.h \
	libmtx/matrix/coordinate/mpi.h \
	libmtx/matrix/coordinate/reorder.h \
	libmtx/matrix/coordinate/sort.h \
	libmtx/mtx/assembly.h \
	libmtx/mtx/blas.h \
	libmtx/mtx/cg.h \
	libmtx/mtx/header.h \
	libmtx/mtx/io.h \
	libmtx/mtx/mpi.h \
	libmtx/mtx/mtx.h \
	libmtx/mtx/precision.h \
	libmtx/mtx/reorder.h \
	libmtx/mtx/sort.h \
	libmtx/mtx/submatrix.h \
	libmtx/mtx/transpose.h \
	libmtx/mtx/triangle.h \
	libmtx/mtxfile/header.h \
	libmtx/mtxfile/comments.h \
	libmtx/mtxfile/mtxfile.h \
	libmtx/mtxfile/size.h \
	libmtx/mtxfile/data.h \
	libmtx/util/index_set.h \
	libmtx/vector/array.h \
	libmtx/vector/array/blas.h \
	libmtx/vector/array/data.h \
	libmtx/vector/array/reorder.h \
	libmtx/vector/coordinate.h \
	libmtx/vector/coordinate/blas.h \
	libmtx/vector/coordinate/data.h \
	libmtx/vector/coordinate/io.h \
	libmtx/vector/coordinate/reorder.h \
	libmtx/version.h

# Header files that will not be distributed
nobase_nodist_include_HEADERS = \
	libmtx/libmtx-config.h

# pkg-config files
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libmtx/libmtx.pc

# Matrix Market utility programs
bin_PROGRAMS = \
	mtxaxpy \
	mtxcg \
	mtxdot \
	mtxgemv \
	mtxinfo \
	mtxnrm2 \
	mtxreorder \
	mtxscal \
	mtxsort
if HAVE_MPI
bin_PROGRAMS += \
	mtxgather \
	mtxscatter
endif
if HAVE_LIBPNG
bin_PROGRAMS += \
	mtxspy
endif

# mtxaxpy
mtxaxpy_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxaxpy_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxaxpy_SOURCES = src/mtxaxpy.c

# mtxcg
mtxcg_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxcg_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxcg_SOURCES = src/mtxcg.c

# mtxdot
mtxdot_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxdot_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxdot_SOURCES = src/mtxdot.c

# mtxgemv
mtxgemv_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxgemv_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxgemv_SOURCES = src/mtxgemv.c

# mtxinfo
mtxinfo_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxinfo_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxinfo_SOURCES = src/mtxinfo.c

# mtxnrm2
mtxnrm2_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxnrm2_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxnrm2_SOURCES = src/mtxnrm2.c

# mtxreorder
mtxreorder_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxreorder_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxreorder_SOURCES = src/mtxreorder.c

# mtxscal
mtxscal_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxscal_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxscal_SOURCES = src/mtxscal.c

# mtxsort
mtxsort_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxsort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxsort_SOURCES = src/mtxsort.c

# mtxgather
mtxgather_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxgather_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxgather_SOURCES = src/mtxgather.c

# mtxscatter
mtxscatter_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxscatter_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxscatter_SOURCES = src/mtxscatter.c

# mtxspy
mtxspy_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS) $(LIBPNG_CPPFLAGS)
mtxspy_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(LIBPNG_LIBS)
mtxspy_SOURCES = src/mtxspy.c

# Unit tests
check_PROGRAMS = \
	tests/test_index_set \
	tests/test_mtx \
	tests/test_mtxfile \
	tests/test_mtx_blas \
	tests/test_mtx_cg \
	tests/test_mtx_io \
	tests/test_mtx_matrix_array \
	tests/test_mtx_matrix_coordinate \
	tests/test_mtx_reorder \
	tests/test_mtx_sort \
	tests/test_mtx_transpose \
	tests/test_parse \
	tests/test_format \
	tests/test_submatrix \
	tests/test_vector

tests_test_index_set_CPPFLAGS = $(MPICPPFLAGS)
tests_test_index_set_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_index_set_SOURCES = tests/test_index_set.c tests/test.h
tests_test_mtx_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_SOURCES = tests/test_mtx.c tests/test.h
tests_test_mtxfile_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxfile_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtxfile_SOURCES = tests/test_mtxfile.c tests/test.h
tests_test_mtx_blas_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_blas_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_blas_SOURCES = tests/test_mtx_blas.c tests/test.h
tests_test_mtx_cg_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_cg_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_cg_SOURCES = tests/test_mtx_cg.c tests/test.h
tests_test_mtx_io_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_io_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_io_SOURCES = tests/test_mtx_io.c tests/test.h
tests_test_mtx_matrix_array_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_matrix_array_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_matrix_array_SOURCES = tests/test_mtx_matrix_array.c tests/test.h
tests_test_mtx_matrix_coordinate_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_matrix_coordinate_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_matrix_coordinate_SOURCES = tests/test_mtx_matrix_coordinate.c tests/test.h
tests_test_mtx_reorder_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_reorder_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_reorder_SOURCES = tests/test_mtx_reorder.c tests/test.h
tests_test_mtx_sort_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_sort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_sort_SOURCES = tests/test_mtx_sort.c tests/test.h
tests_test_mtx_transpose_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_transpose_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_transpose_SOURCES = tests/test_mtx_transpose.c tests/test.h
tests_test_parse_CPPFLAGS = $(MPICPPFLAGS)
tests_test_parse_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_parse_SOURCES = tests/test_parse.c tests/test.h
tests_test_format_CPPFLAGS = $(MPICPPFLAGS)
tests_test_format_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_format_SOURCES = tests/test_format.c tests/test.h
tests_test_submatrix_CPPFLAGS = $(MPICPPFLAGS)
tests_test_submatrix_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_submatrix_SOURCES = tests/test_submatrix.c tests/test.h
tests_test_vector_CPPFLAGS = $(MPICPPFLAGS)
tests_test_vector_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_vector_SOURCES = tests/test_vector.c tests/test.h

TESTS = \
	tests/test_index_set \
	tests/test_mtx \
	tests/test_mtxfile \
	tests/test_mtx_blas \
	tests/test_mtx_cg \
	tests/test_mtx_io \
	tests/test_mtx_matrix_array \
	tests/test_mtx_matrix_coordinate \
	tests/test_mtx_reorder \
	tests/test_mtx_sort \
	tests/test_mtx_transpose \
	tests/test_parse \
	tests/test_format \
	tests/test_submatrix \
	tests/test_vector

# MPI unit tests
if HAVE_MPI
check_PROGRAMS += \
	tests/test_mtx_mpi \
	tests/test_mtxfile_mpi

tests_test_mtx_mpi_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_mpi_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_mpi_SOURCES = tests/test_mtx_mpi.c tests/test.h
tests/test_mtx_mpi_parallel: tests/test_mtx_mpi

tests_test_mtxfile_mpi_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxfile_mpi_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtxfile_mpi_SOURCES = tests/test_mtxfile_mpi.c tests/test.h
tests/test_mtxfile_mpi_parallel: tests/test_mtxfile_mpi

TESTS += \
	tests/test_mtx_mpi_parallel \
	tests/test_mtxfile_mpi_parallel
endif

# Documentation
info_TEXINFOS = manual/libmtx.texi
manual_libmtx_TEXINFOS = \
	manual/introduction.texi \
	manual/installing.texi \
	manual/reporting_bugs.texi \
	manual/matrix_market_file_format.texi \
	manual/matrices_and_vectors.texi \
	manual/distributed_matrices_and_vectors.texi \
	manual/commands.texi \
	manual/c_library_reference.texi \
	manual/references.texi \
	manual/fdl.texi \
	manual/version.texi
AM_MAKEINFOHTMLFLAGS = --css-include=$(srcdir)/manual/texinfo.css
$(top_builddir)/manual/libmtx.html: $(srcdir)/manual/texinfo.css
