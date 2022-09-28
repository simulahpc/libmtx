dnl This file is part of Libmtx
dnl
dnl Copyright (C) 2022 James D. Trotter
dnl
dnl Libmtx is free software: you can redistribute it and/or modify it
dnl under the terms of the GNU General Public License as published by
dnl the Free Software Foundation, either version 3 of the License, or
dnl (at your option) any later version.
dnl
dnl Libmtx is distributed in the hope that it will be useful, but
dnl WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
dnl General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with Libmtx.  If not, see <https://www.gnu.org/licenses/>.
dnl
dnl Authors: James D. Trotter <james@simula.no>
dnl Last modified: 2022-06-28
dnl
dnl autoconf script for the top-level directory of Libmtx.
dnl
dnl Process this file with autoconf to produce a configure script.

AC_INIT([Libmtx],[0.3.0-alpha],[james@simula.no],[libmtx],[https://github.com/simulahpc/libmtx])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([config.h libmtx/libmtx-config.h])

AM_INIT_AUTOMAKE([subdir-objects])

dnl Library versioning (C:R:A == current:revision:age)
dnl See the libtool manual for an explanation of the numbers
dnl
dnl libmtx-0.3.0-alpha	libmtx 1:1:1
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
LIBMTX_CURRENT=1
LIBMTX_REVISION=1
LIBMTX_AGE=1
LIBMTX_LT_VERSION="${LIBMTX_CURRENT}:${LIBMTX_REVISION}:${LIBMTX_AGE}"
AC_SUBST(LIBMTX_LT_VERSION)

dnl Check for programs
AC_LANG([C])
AC_PROG_CC
AC_F77_LIBRARY_LDFLAGS
AM_PROG_AR
LT_INIT

dnl Check for x86 features
AC_CHECK_HEADERS([immintrin.h])
AC_ARG_ENABLE(bmi2,
  AS_HELP_STRING([--disable-bmi2], [disable use of x86 BMI2 instructions]))
AS_IF([test "x$enable_bmi2" != "xno"], [
  AX_GCC_X86_CPU_SUPPORTS(bmi2,
    [AC_DEFINE(LIBMTX_USE_BMI2, 1, [Define to use x86 BMI2 instructions.])
     X86_FEATURE_CFLAGS="$X86_FEATURE_CFLAGS -mbmi2"],
    [if test x"$enable_bmi2" = xyes; then
      AC_MSG_FAILURE([BMI2 not supported.])
     fi])
  ])
AC_SUBST([X86_FEATURE_CFLAGS])
libmtx_CPPFLAGS="$libmtx_CPPFLAGS $X86_FEATURE_CFLAGS"

dnl Check for C math library
AC_SEARCH_LIBS([floor], [m])

dnl Check for MPI
AC_ARG_WITH(mpi,
  [AS_HELP_STRING(
    [--with-mpi],
    [compile with MPI support [default: auto]])],,
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

dnl Check for OpenMP
AC_ARG_WITH(openmp,
  [AS_HELP_STRING(
    [--with-openmp],
    [compile with OpenMP support [default: auto]])],,
  [with_openmp=auto])
openmp_found=no
if test x"$with_openmp" != xno; then
  AX_OPENMP([
    AC_MSG_CHECKING([for openmp])
    openmp_found=yes
    AC_MSG_RESULT([$openmp_found])
    AC_DEFINE(LIBMTX_HAVE_OPENMP,1,[Define to 1 if OpenMP is enabled.])
    AC_SUBST(LIBMTX_HAVE_OPENMP)
    AC_MSG_CHECKING([for openmp compiler flags])
    AC_MSG_RESULT(${OPENMP_CFLAGS})
    AC_SUBST([OPENMP_CFLAGS])
    libmtx_CPPFLAGS="$libmtx_CPPFLAGS ${OPENMP_CFLAGS}"
    LDFLAGS="$LDFLAGS ${OPENMP_CFLAGS}"
   ],[
    AC_MSG_CHECKING([for openmp])
    AC_MSG_RESULT([$openmp_found])
    if test x"$with_openmp" = xyes; then
      AC_MSG_FAILURE([openmp not found.])
    fi])
else
  AC_MSG_CHECKING([for openmp])
  AC_MSG_RESULT([$openmp_found])
fi
AM_CONDITIONAL([HAVE_OPENMP], [test x"$openmp_found" != xno])

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

dnl Check for METIS
AX_LIB_METIS([
  CPPFLAGS="$CPPFLAGS $METIS_INCLUDE"
  LDFLAGS="$LDFLAGS $METIS_LIB"
  LIBS="-lmetis $LIBS"
  libmtx_CPPFLAGS="$libmtx_CPPFLAGS -DLIBMTX_HAVE_METIS"
  AC_DEFINE([LIBMTX_HAVE_METIS], [1], [Define to 1 if you have METIS])
  AC_SUBST(LIBMTX_HAVE_METIS)
],
AC_MSG_WARN([either specify a valid METIS installation with --with-metis=DIR or disable METIS usage with --without-metis]))
AM_CONDITIONAL([HAVE_METIS],
  [test x"$with_metis" != xno -a x"$metis_found" != xno])

dnl check for BLAS libraries
blas_found=no

dnl check for user-defined BLAS library
AC_MSG_CHECKING(for user-defined BLAS library)
AC_ARG_WITH([blas], [
  --with-blas=LIB              base name of user-defined BLAS library (e.g., blas, openblas)
  --with-blas-libdir=DIR       location of user-defined BLAS library
  --with-blas-includedir=DIR   include directory for cblas.h
  --with-blas-ldflags=LDFLAGS  linker flags for user-defined BLAS library
  --without-blas               disable BLAS usage completely])
AC_ARG_WITH([blas-libdir], [])
AC_ARG_WITH([blas-includedir], [])
AC_ARG_WITH([blas-ldflags], [])
AS_IF(
  [test x"$with_blas" == xno -o x"$with_blas" == x], [AC_MSG_RESULT([no])],
  [AC_MSG_RESULT(yes)
   OLD_LDFLAGS="$LDFLAGS"
   OLD_CPPFLAGS="$CPPFLAGS"
   AS_IF([test -n "$with_blas_libdir"], [LDFLAGS="$LDFLAGS -L$with_blas_libdir"])
   AS_IF([test -n "$with_blas_includedir"], [CPPFLAGS="$CPPFLAGS -I$with_blas_includedir"])
   AC_LANG_PUSH([C])
   AC_CHECK_LIB([$with_blas], [cblas_dgemm], [ac_cv_lib_cblas_dgemm=yes], [ac_cv_lib_cblas_dgemm=no], [$with_blas_ldflags])
   AC_CHECK_HEADER([cblas.h], [ac_cv_header_cblas_h=yes], [ac_cv_header_cblas_h=no])
   AC_LANG_POP([C])
   AS_IF([test x"$ac_cv_lib_cblas_dgemm" = x"yes" -a x"$ac_cv_header_cblas_h" = x"yes"],
     [AS_IF([test x"$with_blas_includedir" != x], [BLAS_CPPFLAGS="-I$with_blas_includedir"])
      AS_IF([test x"$with_blas_libdir" != x], [BLAS_LDFLAGS="-L$with_blas_libdir"])
      AS_IF([test x"$with_blas_ldflags" != x], [BLAS_LDFLAGS="$BLAS_LDFLAGS $with_blas_ldflags"])
      BLAS_LIBS="-l$with_blas"
      blas_found=yes])
   LDFLAGS="$OLD_LDFLAGS"
   CPPFLAGS="$OLD_CPPFLAGS"])

dnl check for OpenBLAS
AS_IF([test x"$with_blas" != xno -a x"$blas_found" == xno],
  [AX_OPENBLAS(
    [AC_DEFINE([LIBMTX_HAVE_OPENBLAS], [1], [Define to 1 if you have OpenBLAS])
     AC_SUBST(LIBMTX_HAVE_OPENBLAS)
     BLAS_CPPFLAGS="-I${OPENBLAS_HOME}/include"
     BLAS_LDFLAGS="-L${OPENBLAS_HOME}/lib"
     BLAS_LIBS="-lopenblas"
     with_blas=openblas
     blas_found=yes], [])])
AM_CONDITIONAL([HAVE_OPENBLAS],
  [test x"$with_openblas" != xno -a x"$openblas_found" != xno])

dnl check if any BLAS library was found
AC_MSG_CHECKING([for BLAS])
AS_IF([test x"$with_blas" != xno -a x"$blas_found" != xno],
  [AC_MSG_RESULT([$with_blas])
   AC_SUBST(BLAS_CPPFLAGS)
   AC_SUBST(BLAS_LDFLAGS)
   AC_SUBST(BLAS_LIBS)
   AC_DEFINE([LIBMTX_HAVE_BLAS], [1], [Define to 1 if you have a BLAS library])
   AC_SUBST(LIBMTX_HAVE_BLAS)],
  [AC_MSG_RESULT([no])])
AM_CONDITIONAL([HAVE_BLAS], [test x"$blas_found" != xno])

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
	libmtx/version.h
	libmtx/libmtx-${LIBMTX_MAJOR_VERSION}.pc:libmtx/libmtx.pc.in])
AC_OUTPUT
