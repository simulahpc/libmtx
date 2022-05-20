# This file is part of Libmtx.
#
# Copyright (C) 2022 James D. Trotter
#
# Libmtx is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Libmtx is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
#
# Authors: James D. Trotter <james@simula.no>
# Last modified: 2022-05-19
#
# Makefile for the top-level directory of Libmtx.
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
	manual/image/partition.eps \
	manual/image/partition.pdf \
	manual/image/partition.png \
	manual/image/partition.svg \
	manual/image/webbase-1M.eps \
	manual/image/webbase-1M.png \
	manual/image/webbase-1M_rcm.eps \
	manual/image/webbase-1M_rcm.png \
	manual/texinfo.css \
	tests/test_assumedpartition_parallel \
	tests/test_distsort_parallel \
	tests/test_mtxdistfile_parallel \
	tests/test_mtxfile_mpi_parallel

# Matrix Market library
lib_LTLIBRARIES = libmtx.la
libmtx_la_CPPFLAGS = -I$(top_srcdir) $(MPICPPFLAGS) $(libmtx_CPPFLAGS)
libmtx_la_LDFLAGS = -version-info $(LIBMTX_LT_VERSION) $(MPILDFAGS)
libmtx_la_LIBADD = $(BLAS_LIBS) $(MPILDFLAGS)

# Source files
libmtx_la_SOURCES = \
	libmtx/error.c \
	libmtx/field.c \
	libmtx/matrix/blas.c \
	libmtx/matrix/dense.c \
	libmtx/matrix/dist.c \
	libmtx/matrix/matrix.c \
	libmtx/matrix/matrix_coordinate.c \
	libmtx/matrix/matrix_csr.c \
	libmtx/mtxfile/comments.c \
	libmtx/mtxfile/data.c \
	libmtx/mtxfile/header.c \
	libmtx/mtxfile/mtxdistfile.c \
	libmtx/mtxfile/mtxfile.c \
	libmtx/mtxfile/size.c \
	libmtx/precision.c \
	libmtx/solver/cg.c \
	libmtx/util/cuthill_mckee.c \
	libmtx/util/cuthill_mckee.h \
	libmtx/util/distpartition.c \
	libmtx/util/fmtspec.c \
	libmtx/util/fmtspec.h \
	libmtx/util/index_set.c \
	libmtx/util/io.c \
	libmtx/util/io.h \
	libmtx/util/merge.c \
	libmtx/util/merge.h \
	libmtx/util/parse.c \
	libmtx/util/parse.h \
	libmtx/util/partition.c \
	libmtx/util/permute.c \
	libmtx/util/permute.h \
	libmtx/util/sort.c \
	libmtx/util/sort.h \
	libmtx/matrix/symmetry.c \
	libmtx/matrix/symmetry.h \
	libmtx/matrix/transpose.c \
	libmtx/matrix/transpose.h \
	libmtx/vector/base.c \
	libmtx/vector/blas.c \
	libmtx/vector/dist.c \
	libmtx/vector/omp.c \
	libmtx/vector/packed.c \
	libmtx/vector/vector.c \
	libmtx/version.c

# Header files
nobase_include_HEADERS = \
	libmtx/error.h \
	libmtx/field.h \
	libmtx/libmtx.h \
	libmtx/matrix/blas.h \
	libmtx/matrix/dense.h \
	libmtx/matrix/dist.h \
	libmtx/matrix/matrix.h \
	libmtx/matrix/matrix_coordinate.h \
	libmtx/matrix/matrix_csr.h \
	libmtx/mtxfile/comments.h \
	libmtx/mtxfile/data.h \
	libmtx/mtxfile/header.h \
	libmtx/mtxfile/mtxdistfile.h \
	libmtx/mtxfile/mtxfile.h \
	libmtx/mtxfile/size.h \
	libmtx/precision.h \
	libmtx/solver/cg.h \
	libmtx/util/distpartition.h \
	libmtx/util/index_set.h \
	libmtx/util/partition.h \
	libmtx/vector/base.h \
	libmtx/vector/blas.h \
	libmtx/vector/dist.h \
	libmtx/vector/omp.h \
	libmtx/vector/packed.h \
	libmtx/vector/vector.h \
	libmtx/version.h

# Header files that will not be distributed
nobase_nodist_include_HEADERS = \
	libmtx/libmtx-config.h

# pkg-config files
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libmtx/libmtx-$(LIBMTX_MAJOR_VERSION).pc

#
# Matrix Market utility programs
#
bin_PROGRAMS =

# mtxaxpy
bin_PROGRAMS += mtxaxpy
mtxaxpy_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxaxpy_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxaxpy_SOURCES = src/mtxaxpy.c

# mtxcg
bin_PROGRAMS += mtxcg
mtxcg_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxcg_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxcg_SOURCES = src/mtxcg.c

# mtxdot
bin_PROGRAMS += mtxdot
mtxdot_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxdot_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxdot_SOURCES = src/mtxdot.c

# mtxgemv
bin_PROGRAMS += mtxgemv
mtxgemv_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxgemv_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxgemv_SOURCES = src/mtxgemv.c

# mtxinfo
bin_PROGRAMS += mtxinfo
mtxinfo_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxinfo_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxinfo_SOURCES = src/mtxinfo.c

# mtxnrm2
bin_PROGRAMS += mtxnrm2
mtxnrm2_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxnrm2_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxnrm2_SOURCES = src/mtxnrm2.c

# mtxpartition
bin_PROGRAMS += mtxpartition
mtxpartition_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxpartition_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxpartition_SOURCES = src/mtxpartition.c

# mtxreorder
bin_PROGRAMS += mtxreorder
mtxreorder_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxreorder_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxreorder_SOURCES = src/mtxreorder.c

# mtxscal
bin_PROGRAMS += mtxscal
mtxscal_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxscal_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxscal_SOURCES = src/mtxscal.c

# mtxsort
bin_PROGRAMS += mtxsort
mtxsort_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxsort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxsort_SOURCES = src/mtxsort.c

# mtxtranspose
bin_PROGRAMS += mtxtranspose
mtxtranspose_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxtranspose_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
mtxtranspose_SOURCES = src/mtxtranspose.c

# mtxspy
if HAVE_LIBPNG
bin_PROGRAMS += mtxspy
mtxspy_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS) $(LIBPNG_CPPFLAGS)
mtxspy_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(LIBPNG_LIBS)
mtxspy_SOURCES = src/mtxspy.c
endif

#
# Unit tests
#
check_PROGRAMS = \
	tests/test_index_set \
	tests/test_format \
	tests/test_mtxfile \
	tests/test_mtxmatrix_coordinate \
	tests/test_mtxmatrix_csr \
	tests/test_parse \
	tests/test_partition \
	tests/test_permute \
	tests/test_sort

tests_test_index_set_CPPFLAGS = $(MPICPPFLAGS)
tests_test_index_set_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_index_set_SOURCES = tests/test_index_set.c tests/test.h
tests_test_mtxfile_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxfile_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtxfile_SOURCES = tests/test_mtxfile.c tests/test.h
tests_test_mtxmatrix_coordinate_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxmatrix_coordinate_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtxmatrix_coordinate_SOURCES = tests/test_mtxmatrix_coordinate.c tests/test.h
tests_test_mtxmatrix_csr_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxmatrix_csr_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtxmatrix_csr_SOURCES = tests/test_mtxmatrix_csr.c tests/test.h
tests_test_format_CPPFLAGS = $(MPICPPFLAGS)
tests_test_format_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_format_SOURCES = tests/test_fmtspec.c tests/test.h
tests_test_parse_CPPFLAGS = $(MPICPPFLAGS)
tests_test_parse_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_parse_SOURCES = tests/test_parse.c tests/test.h
tests_test_partition_CPPFLAGS = $(MPICPPFLAGS)
tests_test_partition_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_partition_SOURCES = tests/test_partition.c tests/test.h
tests_test_permute_CPPFLAGS = $(MPICPPFLAGS)
tests_test_permute_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_permute_SOURCES = tests/test_permute.c tests/test.h
tests_test_sort_CPPFLAGS = $(MPICPPFLAGS)
tests_test_sort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_sort_SOURCES = tests/test_sort.c tests/test.h

TESTS = \
	tests/test_format \
	tests/test_index_set \
	tests/test_mtxfile \
	tests/test_mtxmatrix_coordinate \
	tests/test_mtxmatrix_csr \
	tests/test_parse \
	tests/test_partition \
	tests/test_permute \
	tests/test_sort


if HAVE_MPI
tests_test_assumedpartition_CPPFLAGS =
tests_test_assumedpartition_LDADD = $(top_builddir)/libmtx.la
tests_test_assumedpartition_SOURCES = tests/test_assumedpartition.c tests/test.h
check_PROGRAMS += tests/test_assumedpartition
TESTS += tests/test_assumedpartition_parallel
tests/test_assumedpartition_parallel:  tests/test_assumedpartition
endif

if HAVE_MPI
tests_test_mtxdistfile_CPPFLAGS =
tests_test_mtxdistfile_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxdistfile_SOURCES = tests/test_mtxdistfile.c tests/test.h
check_PROGRAMS += tests/test_mtxdistfile
TESTS += tests/test_mtxdistfile_parallel
tests/test_mtxdistfile_parallel:  tests/test_mtxdistfile
endif

tests_test_mtxvector_base_CPPFLAGS =
tests_test_mtxvector_base_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxvector_base_SOURCES = tests/test_mtxvector_base.c tests/test.h
check_PROGRAMS += tests/test_mtxvector_base
TESTS += tests/test_mtxvector_base

if HAVE_BLAS
tests_test_mtxvector_blas_CPPFLAGS =
tests_test_mtxvector_blas_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxvector_blas_SOURCES = tests/test_mtxvector_blas.c tests/test.h
check_PROGRAMS += tests/test_mtxvector_blas
TESTS += tests/test_mtxvector_blas
endif

if HAVE_OPENMP
tests_test_mtxvector_omp_CPPFLAGS =
tests_test_mtxvector_omp_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxvector_omp_SOURCES = tests/test_mtxvector_omp.c tests/test.h
check_PROGRAMS += tests/test_mtxvector_omp
TESTS += tests/test_mtxvector_omp
endif

tests_test_mtxvector_packed_CPPFLAGS =
tests_test_mtxvector_packed_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxvector_packed_SOURCES = tests/test_mtxvector_packed.c tests/test.h
check_PROGRAMS += tests/test_mtxvector_packed
TESTS += tests/test_mtxvector_packed

if HAVE_MPI
tests_test_mtxvector_dist_CPPFLAGS =
tests_test_mtxvector_dist_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxvector_dist_SOURCES = tests/test_mtxvector_dist.c tests/test.h
check_PROGRAMS += tests/test_mtxvector_dist
TESTS += tests/test_mtxvector_dist_parallel
tests/test_mtxvector_dist_parallel:  tests/test_mtxvector_dist
endif

tests_test_mtxmatrix_blas_CPPFLAGS =
tests_test_mtxmatrix_blas_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxmatrix_blas_SOURCES = tests/test_mtxmatrix_blas.c tests/test.h
check_PROGRAMS += tests/test_mtxmatrix_blas
TESTS += tests/test_mtxmatrix_blas

tests_test_mtxmatrix_dense_CPPFLAGS =
tests_test_mtxmatrix_dense_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxmatrix_dense_SOURCES = tests/test_mtxmatrix_dense.c tests/test.h
check_PROGRAMS += tests/test_mtxmatrix_dense
TESTS += tests/test_mtxmatrix_dense

if HAVE_MPI
tests_test_mtxmatrix_dist_CPPFLAGS =
tests_test_mtxmatrix_dist_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxmatrix_dist_SOURCES = tests/test_mtxmatrix_dist.c tests/test.h
check_PROGRAMS += tests/test_mtxmatrix_dist
TESTS += tests/test_mtxmatrix_dist_parallel
tests/test_mtxmatrix_dist_parallel:  tests/test_mtxmatrix_dist
endif

# solver

tests_test_mtxcg_CPPFLAGS =
tests_test_mtxcg_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxcg_SOURCES = tests/test_mtxcg.c tests/test.h
check_PROGRAMS += tests/test_mtxcg
TESTS += tests/test_mtxcg

# util

tests_test_merge_CPPFLAGS =
tests_test_merge_LDADD = $(top_builddir)/libmtx.la
tests_test_merge_SOURCES = tests/test_merge.c tests/test.h
check_PROGRAMS += tests/test_merge
TESTS += tests/test_merge

if HAVE_MPI
tests_test_distpartition_CPPFLAGS =
tests_test_distpartition_LDADD = $(top_builddir)/libmtx.la
tests_test_distpartition_SOURCES = tests/test_distpartition.c tests/test.h
check_PROGRAMS += tests/test_distpartition
TESTS += tests/test_distpartition_parallel
tests/test_distpartition_parallel:  tests/test_distpartition
endif

# MPI unit tests
if HAVE_MPI
check_PROGRAMS += \
	tests/test_distsort \
	tests/test_mtxfile_mpi

tests_test_distsort_CPPFLAGS = $(MPICPPFLAGS)
tests_test_distsort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_distsort_SOURCES = tests/test_distsort.c tests/test.h
tests/test_distsort_parallel: tests/test_distsort

tests_test_mtxfile_mpi_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxfile_mpi_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS)
tests_test_mtxfile_mpi_SOURCES = tests/test_mtxfile_mpi.c tests/test.h
tests/test_mtxfile_mpi_parallel: tests/test_mtxfile_mpi

TESTS += \
	tests/test_distsort_parallel \
	tests/test_mtxfile_mpi_parallel
endif

# Documentation
info_TEXINFOS = manual/libmtx.texi
manual_libmtx_TEXINFOS = \
	manual/introduction.texi \
	manual/installing.texi \
	manual/matrix_market_format.texi \
	manual/matrix_market_files.texi \
	manual/matrices_and_vectors.texi \
	manual/distributed_matrix_market_files.texi \
	manual/distributed_matrices_and_vectors.texi \
	manual/commands.texi \
	manual/references.texi \
	manual/fdl.texi \
	manual/version.texi
AM_MAKEINFOHTMLFLAGS = --css-include=$(srcdir)/manual/texinfo.css
$(top_builddir)/manual/libmtx.html: $(srcdir)/manual/texinfo.css
install-html-local: $(top_srcdir)/manual/image/partition.png
install-html-local: $(top_srcdir)/manual/image/webbase-1M.png
install-html-local: $(top_srcdir)/manual/image/webbase-1M_rcm.png
install-html-local:
	echo " $(MKDIR_P) '$(DESTDIR)$(htmldir)'"; \
	$(INSTALL) -d "$(DESTDIR)$(htmldir)/libmtx.html/image"; \
	echo " $(INSTALL_DATA) -t '$(DESTDIR)$(htmldir)/libmtx.html/image' $(top_srcdir)/manual/image/*"; \
	$(INSTALL_DATA) -t '$(DESTDIR)$(htmldir)/libmtx.html/image' $(top_srcdir)/manual/image/*

uninstall-local:
	-rm -rf "$(DESTDIR)$(htmldir)/libmtx.html/image"
