dnl This file is part of libmtx
dnl
dnl Copyright (C) 2021 James D. Trotter
dnl
dnl libmtx is free software: you can redistribute it and/or
dnl modify it under the terms of the GNU General Public License as
dnl published by the Free Software Foundation, either version 3 of the
dnl License, or (at your option) any later version.
dnl
dnl libmtx is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
dnl General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with libmtx.  If not, see
dnl <https://www.gnu.org/licenses/>.
dnl
dnl Authors: James D. Trotter <james@simula.no>
dnl Last modified: 2021-08-03
dnl
dnl autoconf script for the top-level directory of libmtx.
dnl
dnl Process this file with autoconf to produce a configure script.

AC_INIT([libmtx],[0.1.0],[james@simula.no],[libmtx],[https://github.com/simulahpc/libmtx])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([config.h matrixmarket/libmtx-config.h])

AM_INIT_AUTOMAKE([subdir-objects])

dnl Library versioning (C:R:A == current:revision:age)
dnl See the libtool manual for an explanation of the numbers
dnl
dnl libmtx-0.1.0	libmtx 0:0:0
dnl
dnl How to update library version number
dnl ====================================
dnl
dnl C: increment if the interface has additions, changes or removals.
dnl
dnl R: increment any time the source changes; set to 0 if you
dnl incremented CURRENT
dnl
dnl A: increment if any interfaces have been added; set to 0 if any
dnl interfaces have been removed. removal has precedence over adding,
dnl so set to 0 if both happened.
dnl
dnl See https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
dnl for more detailed info.
LIBMTX_CURRENT=0
LIBMTX_REVISION=0
LIBMTX_AGE=0
LIBMTX_LT_VERSION="${LIBMTX_CURRENT}:${LIBMTX_REVISION}:${LIBMTX_AGE}"
AC_SUBST(LIBMTX_LT_VERSION)

dnl Check for programs
AC_LANG([C])
AC_PROG_CC
AC_F77_LIBRARY_LDFLAGS
AM_PROG_AR
LT_INIT

dnl Check for BLAS
AX_BLAS()
AC_MSG_CHECKING([for BLAS])
if test x"$with_blas" = xno; then
AC_MSG_RESULT([$with_blas])
else
AC_MSG_RESULT([$ax_blas_ok])
if test x"$ax_blas_ok" = xyes; then
AC_DEFINE(LIBMTX_HAVE_BLAS,1,[Define to 1 if you have a BLAS library.])
AC_SUBST(LIBMTX_HAVE_BLAS)
fi
fi

dnl Check for MPI
AC_ARG_WITH(mpi,
  [AS_HELP_STRING(
    [--with-mpi],
    [compile with MPI support. Default: auto])],,
  [with_mpi=auto])
mpi_found=no
if test x"$with_mpi" != xno; then
  AX_MPI([
    AC_MSG_CHECKING([for MPI])
    mpi_found=yes
    AC_MSG_RESULT([$mpi_found])
    AC_DEFINE(LIBMTX_HAVE_MPI,1,[Define to 1 if you have an MPI library.])
    AC_SUBST(LIBMTX_HAVE_MPI)
    AC_MSG_CHECKING([for mpicc compiler flags])
    MPICPPFLAGS="`${MPICC} --showme:compile`"
    AC_MSG_RESULT(${MPICPPFLAGS})
    AC_SUBST([MPICPPFLAGS])
    AC_MSG_CHECKING([for mpicc linker flags])
    MPILDFLAGS="`${MPICC} --showme:link`"
    AC_MSG_RESULT(${MPILDFLAGS})
    AC_SUBST([MPILDFLAGS])
    libmtx_CPPFLAGS="$libmtx_CPPFLAGS -DLIBMTX_HAVE_MPI"
   ],[
    AC_MSG_CHECKING([for MPI])
    AC_MSG_RESULT([$mpi_found])
    if test x"$with_mpi" = xyes; then
      AC_MSG_FAILURE([MPI not found.])
    fi])
else
  AC_MSG_CHECKING([for MPI])
  AC_MSG_RESULT([$mpi_found])
fi
AM_CONDITIONAL([HAVE_MPI], [test x"$mpi_found" != xno])

dnl Check for zlib
AX_CHECK_ZLIB([
  CPPFLAGS="$CPPFLAGS -I${ZLIB_HOME}/include"
  LDFLAGS="$LDFLAGS -L${ZLIB_HOME}/lib"
  LIBS="-lz $LIBS"
  libmtx_CPPFLAGS="$libmtx_CPPFLAGS -DLIBMTX_HAVE_LIBZ"
  AC_DEFINE([LIBMTX_HAVE_LIBZ], [1],
    [Define to 1 if you have `z' library (-lz)])
  AC_SUBST(LIBMTX_HAVE_LIBZ)
])

dnl Check for libpng
AX_LIBPNG([
  LIBPNG_CPPFLAGS="-I${LIBPNG_HOME}/include"
  LIBPNG_LDFLAGS="-L${LIBPNG_HOME}/lib"
  LIBPNG_LIBS="-lpng"
  AC_SUBST(LIBPNG_CPPFLAGS)
  AC_SUBST(LIBPNG_LDFLAGS)
  AC_SUBST(LIBPNG_LIBS)
  AC_DEFINE([LIBMTX_HAVE_LIBPNG], [1], [Define to 1 if you have libpng])
  AC_SUBST(LIBMTX_HAVE_LIBPNG)
])
AM_CONDITIONAL([HAVE_LIBPNG],
  [test x"$with_libpng" != xno -a x"$libpng_found" != xno])

AC_SUBST(libmtx_CPPFLAGS)

dnl Check for SuperLU_DIST
AX_SUPERLU_DIST([
  SUPERLU_DIST_CPPFLAGS="-I${SUPERLU_DIST_HOME}/include"
  SUPERLU_DIST_LDFLAGS="-L${SUPERLU_DIST_HOME}/lib"
  SUPERLU_DIST_LIBS="-lsuperlu_dist"
  AC_SUBST(SUPERLU_DIST_CPPFLAGS)
  AC_SUBST(SUPERLU_DIST_LDFLAGS)
  AC_SUBST(SUPERLU_DIST_LIBS)
  AC_DEFINE([LIBMTX_HAVE_SUPERLU_DIST], [1], [Define to 1 if you have SuperLU_DIST])
  AC_SUBST(LIBMTX_HAVE_SUPERLU_DIST)
])
AM_CONDITIONAL([HAVE_SUPERLU_DIST],
  [test x"$with_superlu_dist" != xno -a x"$superlu_dist_found" != xno])

AC_SUBST(libmtx_CPPFLAGS)

dnl Version information - split VERSION into major, minor and release
AC_PROG_SED
LIBMTX_VERSION="$VERSION"
LIBMTX_MAJOR_VERSION=`echo "$VERSION" | $SED 's/\([[^.]][[^.]]*\).*/\1/'`
LIBMTX_MINOR_VERSION=`echo "$VERSION" | $SED 's/[[^.]][[^.]]*.\([[^.]][[^.]]*\).*/\1/'`
LIBMTX_RELEASE_VERSION=`echo "$VERSION" | $SED 's/[[^.]][[^.]]*.[[^.]][[^.]]*.\([[^.]][[^.]]*\).*/\1/'`
AC_SUBST(LIBMTX_VERSION)
AC_SUBST(LIBMTX_MAJOR_VERSION)
AC_SUBST(LIBMTX_MINOR_VERSION)
AC_SUBST(LIBMTX_RELEASE_VERSION)

AC_CONFIG_FILES([
	Makefile
	matrixmarket/version.h
	matrixmarket/libmtx.pc:matrixmarket/libmtx.pc.in])
AC_OUTPUT
