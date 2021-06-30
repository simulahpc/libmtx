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
# Last modified: 2021-06-18
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
	matrixmarket/blas.c \
	matrixmarket/error.c \
	matrixmarket/header.c \
	matrixmarket/index_set.c \
	matrixmarket/io.c \
	matrixmarket/matrix.c \
	matrixmarket/matrix_array.c \
	matrixmarket/matrix_coordinate.c \
	matrixmarket/mpi.c \
	matrixmarket/mtx.c \
	matrixmarket/parse.c \
	matrixmarket/parse.h \
	matrixmarket/format.c \
	matrixmarket/format.h \
	matrixmarket/sort.c \
	matrixmarket/submatrix.c \
	matrixmarket/transpose.c \
	matrixmarket/vector_array.c \
	matrixmarket/vector_coordinate.c \
	matrixmarket/version.c

# Header files
nobase_include_HEADERS = \
	matrixmarket/blas.h \
	matrixmarket/error.h \
	matrixmarket/header.h \
	matrixmarket/index_set.h \
	matrixmarket/io.h \
	matrixmarket/matrix.h \
	matrixmarket/matrix_array.h \
	matrixmarket/matrix_coordinate.h \
	matrixmarket/matrixmarket.h \
	matrixmarket/mpi.h \
	matrixmarket/mtx.h \
	matrixmarket/vector_array.h \
	matrixmarket/vector_coordinate.h \
	matrixmarket/version.h

# Header files that will not be distributed
nobase_nodist_include_HEADERS = \
	matrixmarket/libmtx-config.h

# pkg-config files
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = matrixmarket/libmtx.pc

# Matrix Market utility programs
bin_PROGRAMS = \
	mtxinfo \
	mtxgemv \
	mtxsort
if HAVE_MPI
bin_PROGRAMS += \
	mtxgather \
	mtxscatter
endif

# mtxinfo
mtxinfo_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxinfo_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxinfo_SOURCES = src/mtxinfo.c

# mtxgemv
mtxgemv_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxgemv_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxgemv_SOURCES = src/mtxgemv.c

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

# Unit tests
check_PROGRAMS = \
	tests/test_index_set \
	tests/test_mtx \
	tests/test_mtx_blas \
	tests/test_mtx_io \
	tests/test_mtx_matrix_array \
	tests/test_mtx_matrix_coordinate \
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
tests_test_mtx_blas_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_blas_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_blas_SOURCES = tests/test_mtx_blas.c tests/test.h
tests_test_mtx_io_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_io_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_io_SOURCES = tests/test_mtx_io.c tests/test.h
tests_test_mtx_matrix_array_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_matrix_array_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_matrix_array_SOURCES = tests/test_mtx_matrix_array.c tests/test.h
tests_test_mtx_matrix_coordinate_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_matrix_coordinate_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_matrix_coordinate_SOURCES = tests/test_mtx_matrix_coordinate.c tests/test.h
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
	tests/test_mtx_blas \
	tests/test_mtx_io \
	tests/test_mtx_matrix_array \
	tests/test_mtx_matrix_coordinate \
	tests/test_mtx_sort \
	tests/test_mtx_transpose \
	tests/test_parse \
	tests/test_format \
	tests/test_submatrix \
	tests/test_vector

# MPI unit tests
if HAVE_MPI
check_PROGRAMS += \
	tests/test_mtx_mpi

tests_test_mtx_mpi_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtx_mpi_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtx_mpi_SOURCES = tests/test_mtx_mpi.c tests/test.h
tests/test_mtx_mpi_parallel: tests/test_mtx_mpi

TESTS += \
	tests/test_mtx_mpi_parallel
endif

# Documentation
info_TEXINFOS = manual/libmtx.texi
manual_libmtx_TEXINFOS = \
	manual/introduction.texi \
	manual/installing.texi \
	manual/reporting_bugs.texi \
	manual/matrix_market_file_format.texi \
	manual/matrix_market_objects.texi \
	manual/commands.texi \
	manual/c_library_reference.texi \
	manual/references.texi \
	manual/fdl.texi \
	manual/version.texi
AM_MAKEINFOHTMLFLAGS = --css-include=$(srcdir)/manual/texinfo.css
$(top_builddir)/manual/libmtx.html: $(srcdir)/manual/texinfo.css
