# ax_superlu_dist.m4
#
# SYNOPSIS
#
#   AX_SUPERLU_DIST([action-if-found], [action-if-not-found])
#
# DESCRIPTION
#
#   This macro searches for an installed SuperLU_DIST library. If nothing was
#   specified when calling configure, it searches first in /usr/local and
#   then in /usr, /opt/local and /sw. If the --with-superlu_dist=DIR is specified,
#   it will try to find it in DIR/include/superlu_defs.h and DIR/lib/superlu_dist.a. If
#   --without-superlu_dist is specified, the library is not searched for at all.
#
#   If either the header file (superlu_defs.h) or the library (superlu_dist) is not found,
#   shell commands 'action-if-not-found' is run. If 'action-if-not-found' is
#   not specified, the configuration exits on error, asking for a valid SuperLU_DIST
#   installation directory or --without-superlu_dist.
#
#   If both header file and library are found, shell commands 'action-if-found'
#   is run. If 'action-if-found' is not specified, the default action sets
#   SUPERLU_DIST_CPPFLAGS to '-I${SUPERLU_DIST_HOME}/include', SUPERLU_DIST_LDFLAGS to
#   '-L${SUPERLU_DIST_HOME}/lib', and SUPERLU_DIST_LIBS to '-lsuperlu_dist', and calls
#   AC_DEFINE(HAVE_SUPERLU_DIST). You should use autoheader to include a definition
#   for this symbol in a config.h file. Sample usage in a C/C++ source
#   is as follows:
#
#     #ifdef HAVE_SUPERLU_DIST
#     #include <superlu_defs.h>
#     #endif /* HAVE_SUPERLU_DIST */
#
# LICENSE
#
#   Copyright (c) 2021 James D. Trotter <james@simula.no>
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

AU_ALIAS([SUPERLU_DIST], [AX_SUPERLU_DIST])
AC_DEFUN([AX_SUPERLU_DIST],
#
# Handle user hints
#
[AC_MSG_CHECKING(if SuperLU_DIST is wanted)
superlu_dist_places="/usr/local /usr /opt/local /sw"
AC_ARG_WITH([superlu_dist],
[  --with-superlu_dist=DIR         path to SuperLU_DIST installation @<:@defaults to
                          /usr/local or /usr if not found in /usr/local@:>@
  --without-superlu_dist          disable SuperLU_DIST usage completely],
[if test "$withval" != no ; then
  with_superlu_dist="$withval"
  AC_MSG_RESULT(yes)
  if test -d "$withval"
  then
    superlu_dist_places="$withval $superlu_dist_places"
  else
    AC_MSG_WARN([Sorry, $withval does not exist, checking usual places])
  fi
else
  with_superlu_dist=no
  superlu_dist_places=
  AC_MSG_RESULT(no)
fi],
[AC_MSG_RESULT(yes)])

#
# Locate Superlu_DIST, if wanted
#
superlu_dist_found=no
if test -n "${superlu_dist_places}"
then
	# check the user supplied or any other more or less 'standard' place:
	#   Most UNIX systems      : /usr/local and /usr
	#   MacPorts / Fink on OSX : /opt/local respectively /sw
	for SUPERLU_DIST_HOME in ${superlu_dist_places} ; do
	  if test -f "${SUPERLU_DIST_HOME}/include/superlu_defs.h"; then break; fi
	  SUPERLU_DIST_HOME=""
	done

  SUPERLU_DIST_OLD_LDFLAGS=$LDFLAGS
  SUPERLU_DIST_OLD_CPPFLAGS=$CPPFLAGS
  if test -n "${SUPERLU_DIST_HOME}"; then
        LDFLAGS="$LDFLAGS -L${SUPERLU_DIST_HOME}/lib"
        CPPFLAGS="$CPPFLAGS -I${SUPERLU_DIST_HOME}/include"
  fi
  AC_LANG_PUSH([C])
  AC_CHECK_LIB([superlu_dist], [pdgssvx], [superlu_dist_cv_superlu_dist=yes], [superlu_dist_cv_superlu_dist=no])
  AC_CHECK_HEADER([superlu_defs.h], [superlu_dist_cv_superlu_defs_h=yes], [superlu_dist_cv_superlu_defs_h=no])
  AC_LANG_POP([C])
  if test "$superlu_dist_cv_superlu_dist" = "yes" && test "$superlu_dist_cv_superlu_defs_h" = "yes"
  then
    #
    # If both library and header were found, action-if-found
    #
    superlu_dist_found=yes
    m4_ifblank([$1],[
                SUPERLU_DIST_CPPFLAGS="-I${SUPERLU_DIST_HOME}/include"
                SUPERLU_DIST_LDFLAGS="-L${SUPERLU_DIST_HOME}/lib"
                SUPERLU_DIST_LIBS="-lsuperlu_dist"
                AC_DEFINE([HAVE_SUPERLU_DIST], [1],
                          [Define to 1 if you have superlu_dist])
               ],[
                # Restore variables
                LDFLAGS="$SUPERLU_DIST_OLD_LDFLAGS"
                CPPFLAGS="$SUPERLU_DIST_OLD_CPPFLAGS"
                $1
               ])
  else
    #
    # If either header or library was not found, action-if-not-found
    #
    m4_default([$2],[
                AC_MSG_ERROR([either specify a valid SuperLU_DIST installation with --with-superlu_dist=DIR or disable SuperLU_DIST usage with --without-superlu_dist])
                ])
  fi
fi
])
