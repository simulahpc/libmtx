# ax_openblas.m4
#
# SYNOPSIS
#
#   AX_OPENBLAS([action-if-found], [action-if-not-found])
#
# DESCRIPTION
#
#   This macro searches for an installed OpenBLAS library, possibly in
#   a user-specified location. The user may specify the location of
#   the installed library by using the --with-openblas option when
#   calling configure. If no location is given, then configure will
#   /usr/local and then /usr.  If the option --with-openblas=DIR is
#   specified, then configure will try to find DIR/include/cblas.h and
#   DIR/lib/openblas.a. If --without-openblas is specified, the
#   library is not searched for at all.
#
#   If either the CBLAS header file (cblas.h) or the library
#   (openblas) is not found, shell commands 'action-if-not-found' is
#   run. If 'action-if-not-found' is not specified, the configuration
#   exits on error, asking for a valid OpenBLAS installation directory
#   or --without-openblas.
#
#   If both header file and library are found, shell commands
#   'action-if-found' is run. If 'action-if-found' is not specified,
#   the default action sets OPENBLAS_CPPFLAGS to
#   '-I${OPENBLAS_HOME}/include', OPENBLAS_LDFLAGS to
#   '-L${OPENBLAS_HOME}/lib', and OPENBLAS_LIBS to '-lopenblas', and
#   calls AC_DEFINE(HAVE_OPENBLAS). You should use autoheader to
#   include a definition for this symbol in a config.h file. Sample
#   usage in a C/C++ source is as follows:
#
#     #ifdef HAVE_OPENBLAS
#     #include <cblas.h>
#     #endif /* HAVE_OPENBLAS */
#
# LICENSE
#
#   Copyright (c) 2022 James D. Trotter <james@simula.no>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <https://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AU_ALIAS([OPENBLAS], [AX_OPENBLAS])
AC_DEFUN([AX_OPENBLAS],
#
# Handle user hints
#
[AC_MSG_CHECKING(if OpenBLAS is wanted)
openblas_places="/usr/local /usr"
AC_ARG_WITH([openblas],
[  --with-openblas=DIR     path to OpenBLAS installation @<:@defaults to
                          /usr/local or /usr if not found in /usr/local@:>@
  --without-openblas      disable OpenBLAS usage completely],
[if test "$withval" != no ; then
  with_openblas="$withval"
  AC_MSG_RESULT(yes)
  if test -d "$withval"
  then
    openblas_places="$withval $openblas_places"
  else
    AC_MSG_WARN([Sorry, $withval does not exist, checking usual places])
  fi
else
  with_openblas=no
  openblas_places=
  AC_MSG_RESULT(no)
fi],
[AC_MSG_RESULT(yes)])

#
# Locate openblas, if wanted
#
openblas_found=no
if test -n "${openblas_places}"
then
	# check the user supplied or any other more or less 'standard' place:
	#   Most UNIX systems      : /usr/local and /usr
	for OPENBLAS_HOME in ${openblas_places} ; do
	  if test -f "${OPENBLAS_HOME}/include/cblas.h"; then break; fi
	  OPENBLAS_HOME=""
	done

  OPENBLAS_OLD_LDFLAGS=$LDFLAGS
  OPENBLAS_OLD_CPPFLAGS=$CPPFLAGS
  if test -n "${OPENBLAS_HOME}"; then
        LDFLAGS="$LDFLAGS -L${OPENBLAS_HOME}/lib"
        CPPFLAGS="$CPPFLAGS -I${OPENBLAS_HOME}/include"
  fi
  AC_LANG_PUSH([C])
  AC_CHECK_LIB([openblas], [cblas_dgemm], [openblas_cv_openblas=yes], [openblas_cv_openblas=no])
  AC_CHECK_HEADER([cblas.h], [openblas_cv_cblas_h=yes], [openblas_cv_cblas_h=no])
  AC_LANG_POP([C])
  if test "$openblas_cv_openblas" = "yes" && test "$openblas_cv_cblas_h" = "yes"
  then
    #
    # If both library and header were found, action-if-found
    #
    openblas_found=yes
    m4_ifblank([$1],[
                OPENBLAS_CPPFLAGS="-I${OPENBLAS_HOME}/include"
                OPENBLAS_LDFLAGS="-L${OPENBLAS_HOME}/lib"
                OPENBLAS_LIBS="-lopenblas"
                AC_DEFINE([HAVE_OPENBLAS], [1],
                          [Define to 1 if you have openblas])
               ],[
                # Restore variables
                LDFLAGS="$OPENBLAS_OLD_LDFLAGS"
                CPPFLAGS="$OPENBLAS_OLD_CPPFLAGS"
                $1
               ])
  else
    #
    # If either header or library was not found, action-if-not-found
    #
    m4_default([$2],[
                AC_MSG_ERROR([either specify a valid openblas installation with --with-openblas=DIR or disable openblas usage with --without-openblas])
                ])
  fi
fi
])
