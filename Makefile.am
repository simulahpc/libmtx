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
# Last modified: 2022-10-10
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
	tests/test_mtxfile_mpi_parallel \
	tests/test_mtxmpimatrix_parallel \
	tests/test_mtxmpivector_parallel

# Matrix Market library
lib_LTLIBRARIES = libmtx.la
libmtx_la_CPPFLAGS = -I$(top_srcdir) @CFLAG_VISIBILITY@ -DLIBMTX_API_EXPORT @BMI2_CFLAGS@ $(MPICPPFLAGS) $(BLAS_CPPFLAGS) $(libmtx_CPPFLAGS)
libmtx_la_LDFLAGS = -version-info $(LIBMTX_LT_VERSION) $(BLAS_LDFLAGS) $(MPILDFAGS)
libmtx_la_LIBADD = $(BLAS_LIBS) $(BLAS_LDFLAGS) $(MPILDFLAGS)

# Source files
libmtx_la_SOURCES = \
	libmtx/error.c \
	libmtx/linalg/base/coo.c \
	libmtx/linalg/base/csr.c \
	libmtx/linalg/base/dense.c \
	libmtx/linalg/base/vector.c \
	libmtx/linalg/blas/dense.c \
	libmtx/linalg/blas/vector.c \
	libmtx/linalg/field.c \
	libmtx/linalg/gemvoverlap.c \
	libmtx/linalg/local/matrix.c \
	libmtx/linalg/local/vector.c \
	libmtx/linalg/mpi/matrix.c \
	libmtx/linalg/mpi/vector.c \
	libmtx/linalg/null/coo.c \
	libmtx/linalg/null/vector.c \
	libmtx/linalg/omp/csr.c \
	libmtx/linalg/omp/vector.c \
	libmtx/linalg/partition.c \
	libmtx/linalg/precision.c \
	libmtx/linalg/symmetry.c \
	libmtx/linalg/transpose.c \
	libmtx/mtxfile/comments.c \
	libmtx/mtxfile/data.c \
	libmtx/mtxfile/header.c \
	libmtx/mtxfile/mtxdistfile.c \
	libmtx/mtxfile/mtxfile.c \
	libmtx/mtxfile/size.c \
	libmtx/solver/cg.c \
	libmtx/util/cuthill_mckee.c \
	libmtx/util/cuthill_mckee.h \
	libmtx/util/fmtspec.c \
	libmtx/util/fmtspec.h \
	libmtx/util/merge.c \
	libmtx/util/merge.h \
	libmtx/util/metis.c \
	libmtx/util/metis.h \
	libmtx/util/morton.c \
	libmtx/util/morton.h \
	libmtx/util/mpipartition.c \
	libmtx/util/mpipartition.h \
	libmtx/util/mpisort.c \
	libmtx/util/mpisort.h \
	libmtx/util/partition.c \
	libmtx/util/partition.h \
	libmtx/util/permute.c \
	libmtx/util/permute.h \
	libmtx/util/sort.c \
	libmtx/util/sort.h \
	libmtx/version.c

# Header files
nobase_include_HEADERS = \
	libmtx/error.h \
	libmtx/libmtx.h \
	libmtx/linalg/base/coo.h \
	libmtx/linalg/base/csr.h \
	libmtx/linalg/base/dense.h \
	libmtx/linalg/base/vector.h \
	libmtx/linalg/blas/dense.h \
	libmtx/linalg/blas/vector.h \
	libmtx/linalg/field.h \
	libmtx/linalg/gemvoverlap.h \
	libmtx/linalg/local/matrix.h \
	libmtx/linalg/local/vector.h \
	libmtx/linalg/mpi/matrix.h \
	libmtx/linalg/mpi/vector.h \
	libmtx/linalg/null/coo.h \
	libmtx/linalg/null/vector.h \
	libmtx/linalg/omp/csr.h \
	libmtx/linalg/omp/vector.h \
	libmtx/linalg/partition.h \
	libmtx/linalg/precision.h \
	libmtx/linalg/symmetry.h \
	libmtx/linalg/transpose.h \
	libmtx/mtxfile/comments.h \
	libmtx/mtxfile/data.h \
	libmtx/mtxfile/header.h \
	libmtx/mtxfile/mtxdistfile.h \
	libmtx/mtxfile/mtxfile.h \
	libmtx/mtxfile/size.h \
	libmtx/solver/cg.h \
	libmtx/version.h

# Header files that will not be distributed
nobase_nodist_include_HEADERS = \
	libmtx/libmtx-config.h

# pkg-config files
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libmtx/libmtx-$(LIBMTX_MAJOR_VERSION).pc

#
# utility programs
#
bin_PROGRAMS =

# mtxaxpy
bin_PROGRAMS += mtxaxpy
mtxaxpy_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxaxpy_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxaxpy_SOURCES = src/mtxaxpy.c src/parse.c src/parse.h

# mtxcg
bin_PROGRAMS += mtxcg
mtxcg_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxcg_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxcg_SOURCES = src/mtxcg.c src/parse.c src/parse.h

# mtxdot
bin_PROGRAMS += mtxdot
mtxdot_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxdot_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxdot_SOURCES = src/mtxdot.c src/parse.c src/parse.h

# mtxgemm
bin_PROGRAMS += mtxgemm
mtxgemm_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxgemm_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxgemm_SOURCES = src/mtxgemm.c src/parse.c src/parse.h

# mtxgemv
bin_PROGRAMS += mtxgemv
mtxgemv_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxgemv_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxgemv_SOURCES = src/mtxgemv.c src/parse.c src/parse.h

# mtxinfo
bin_PROGRAMS += mtxinfo
mtxinfo_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxinfo_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxinfo_SOURCES = src/mtxinfo.c src/parse.c src/parse.h

# mtxnrm2
bin_PROGRAMS += mtxnrm2
mtxnrm2_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxnrm2_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxnrm2_SOURCES = src/mtxnrm2.c src/parse.c src/parse.h

# mtxpartition
bin_PROGRAMS += mtxpartition
mtxpartition_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxpartition_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxpartition_SOURCES = src/mtxpartition.c src/parse.c src/parse.h

# mtxrand
bin_PROGRAMS += mtxrand
mtxrand_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxrand_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxrand_SOURCES = src/mtxrand.c src/parse.c src/parse.h

# mtxreorder
bin_PROGRAMS += mtxreorder
mtxreorder_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxreorder_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxreorder_SOURCES = src/mtxreorder.c src/parse.c src/parse.h

# mtxsplit
bin_PROGRAMS += mtxsplit
mtxsplit_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxsplit_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxsplit_SOURCES = src/mtxsplit.c src/parse.c src/parse.h

# mtxscal
bin_PROGRAMS += mtxscal
mtxscal_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxscal_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxscal_SOURCES = src/mtxscal.c src/parse.c src/parse.h

# mtxsort
bin_PROGRAMS += mtxsort
mtxsort_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxsort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxsort_SOURCES = src/mtxsort.c src/parse.c src/parse.h

# mtxtranspose
bin_PROGRAMS += mtxtranspose
mtxtranspose_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS)
mtxtranspose_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
mtxtranspose_SOURCES = src/mtxtranspose.c src/parse.c src/parse.h

# mtxspy
if HAVE_LIBPNG
bin_PROGRAMS += mtxspy
mtxspy_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src $(MPICPPFLAGS) $(LIBPNG_CPPFLAGS)
mtxspy_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS) $(LIBPNG_LIBS)
mtxspy_SOURCES = src/mtxspy.c src/parse.c src/parse.h
endif

#
# Unit tests
#
check_PROGRAMS =
TESTS =

# mtxfile
check_PROGRAMS += tests/test_mtxfile
tests_test_mtxfile_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxfile_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_mtxfile_SOURCES = tests/test_mtxfile.c tests/test.h
TESTS += tests/test_mtxfile
if HAVE_MPI
tests_test_mtxfile_mpi_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxfile_mpi_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_mtxfile_mpi_SOURCES = tests/test_mtxfile_mpi.c tests/test.h
check_PROGRAMS += tests/test_mtxfile_mpi
TESTS += tests/test_mtxfile_mpi_parallel
tests/test_mtxfile_mpi_parallel: tests/test_mtxfile_mpi
tests_test_mtxdistfile_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxdistfile_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_mtxdistfile_SOURCES = tests/test_mtxdistfile.c tests/test.h
check_PROGRAMS += tests/test_mtxdistfile
TESTS += tests/test_mtxdistfile_parallel
tests/test_mtxdistfile_parallel:  tests/test_mtxdistfile
endif

# vector
tests_test_mtxbasevector_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxbasevector_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxbasevector_SOURCES = tests/test_mtxbasevector.c tests/test.h
check_PROGRAMS += tests/test_mtxbasevector
TESTS += tests/test_mtxbasevector

if HAVE_BLAS
tests_test_mtxblasvector_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxblasvector_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxblasvector_SOURCES = tests/test_mtxblasvector.c tests/test.h
check_PROGRAMS += tests/test_mtxblasvector
TESTS += tests/test_mtxblasvector
endif

if HAVE_OPENMP
tests_test_mtxompvector_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxompvector_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxompvector_SOURCES = tests/test_mtxompvector.c tests/test.h
check_PROGRAMS += tests/test_mtxompvector
TESTS += tests/test_mtxompvector
endif

if HAVE_MPI
tests_test_mtxmpivector_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxmpivector_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_mtxmpivector_SOURCES = tests/test_mtxmpivector.c tests/test.h
check_PROGRAMS += tests/test_mtxmpivector
TESTS += tests/test_mtxmpivector_parallel
tests/test_mtxmpivector_parallel:  tests/test_mtxmpivector
endif

# matrix
tests_test_mtxbasecoo_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxbasecoo_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxbasecoo_SOURCES = tests/test_mtxbasecoo.c tests/test.h
check_PROGRAMS += tests/test_mtxbasecoo
TESTS += tests/test_mtxbasecoo
tests_test_mtxbasecsr_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxbasecsr_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxbasecsr_SOURCES = tests/test_mtxbasecsr.c tests/test.h
check_PROGRAMS += tests/test_mtxbasecsr
TESTS += tests/test_mtxbasecsr
tests_test_mtxbasedense_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxbasedense_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxbasedense_SOURCES = tests/test_mtxbasedense.c tests/test.h
check_PROGRAMS += tests/test_mtxbasedense
TESTS += tests/test_mtxbasedense
if HAVE_BLAS
tests_test_mtxblasdense_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxblasdense_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxblasdense_SOURCES = tests/test_mtxblasdense.c tests/test.h
check_PROGRAMS += tests/test_mtxblasdense
TESTS += tests/test_mtxblasdense
endif
if HAVE_OPENMP
tests_test_mtxompcsr_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxompcsr_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxompcsr_SOURCES = tests/test_mtxompcsr.c tests/test.h
check_PROGRAMS += tests/test_mtxompcsr
TESTS += tests/test_mtxompcsr
endif
if HAVE_MPI
tests_test_mtxmpimatrix_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxmpimatrix_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_mtxmpimatrix_SOURCES = tests/test_mtxmpimatrix.c tests/test.h
check_PROGRAMS += tests/test_mtxmpimatrix
TESTS += tests/test_mtxmpimatrix_parallel
tests/test_mtxmpimatrix_parallel:  tests/test_mtxmpimatrix
endif

# solver
tests_test_mtxcg_CPPFLAGS = $(MPICPPFLAGS)
tests_test_mtxcg_LDADD = $(top_builddir)/libmtx.la
tests_test_mtxcg_SOURCES = tests/test_mtxcg.c tests/test.h
check_PROGRAMS += tests/test_mtxcg
TESTS += tests/test_mtxcg

# util
tests_test_fmtspec_CPPFLAGS =
tests_test_fmtspec_LDADD =
tests_test_fmtspec_SOURCES = tests/test_fmtspec.c tests/test.h libmtx/util/fmtspec.c
check_PROGRAMS += tests/test_fmtspec
TESTS += tests/test_fmtspec
tests_test_merge_CPPFLAGS =
tests_test_merge_LDADD =
tests_test_merge_SOURCES = tests/test_merge.c tests/test.h libmtx/util/merge.c libmtx/util/sort.c
check_PROGRAMS += tests/test_merge
TESTS += tests/test_merge
tests_test_morton_CPPFLAGS = @BMI2_CFLAGS@
tests_test_morton_LDADD =
tests_test_morton_SOURCES = tests/test_morton.c tests/test.h libmtx/util/morton.c
check_PROGRAMS += tests/test_morton
TESTS += tests/test_morton
tests_test_parse_CPPFLAGS =
tests_test_parse_LDADD =
tests_test_parse_SOURCES = tests/test_parse.c tests/test.h src/parse.c src/parse.h
check_PROGRAMS += tests/test_parse
TESTS += tests/test_parse
tests_test_partition_CPPFLAGS =
tests_test_partition_LDADD =
tests_test_partition_SOURCES = tests/test_partition.c tests/test.h libmtx/util/partition.c
check_PROGRAMS += tests/test_partition
TESTS += tests/test_partition
tests_test_permute_CPPFLAGS =
tests_test_permute_LDADD =
tests_test_permute_SOURCES = tests/test_permute.c tests/test.h libmtx/util/permute.c
check_PROGRAMS += tests/test_permute
TESTS += tests/test_permute
tests_test_sort_CPPFLAGS =
tests_test_sort_LDADD =
tests_test_sort_SOURCES = tests/test_sort.c tests/test.h libmtx/util/sort.c
check_PROGRAMS += tests/test_sort
TESTS += tests/test_sort
if HAVE_MPI
tests_test_assumedpartition_CPPFLAGS = $(MPICPPFLAGS)
tests_test_assumedpartition_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_assumedpartition_SOURCES = tests/test_assumedpartition.c tests/test.h libmtx/util/mpipartition.c libmtx/util/partition.c libmtx/util/sort.c
check_PROGRAMS += tests/test_assumedpartition
TESTS += tests/test_assumedpartition_parallel
tests/test_assumedpartition_parallel: tests/test_assumedpartition
tests_test_distsort_CPPFLAGS = $(MPICPPFLAGS)
tests_test_distsort_LDADD = $(top_builddir)/libmtx.la $(MPILDFLAGS) $(BLAS_LIBS) $(BLAS_LDFLAGS)
tests_test_distsort_SOURCES = tests/test_distsort.c tests/test.h libmtx/util/mpisort.c libmtx/util/sort.c
check_PROGRAMS += tests/test_distsort
TESTS += tests/test_distsort_parallel
tests/test_distsort_parallel: tests/test_distsort
endif

#
# Documentation
#
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
