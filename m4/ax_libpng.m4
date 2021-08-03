# ax_libpng.m4
#
# SYNOPSIS
#
#   AX_LIBPNG([action-if-found], [action-if-not-found])
#
# DESCRIPTION
#
#   This macro searches for an installed libpng library. If nothing was
#   specified when calling configure, it searches first in /usr/local and
#   then in /usr, /opt/local and /sw. If the --with-libpng=DIR is specified,
#   it will try to find it in DIR/include/png.h and DIR/lib/libpng.a. If
#   --without-libpng is specified, the library is not searched for at all.
#
#   If either the header file (png.h) or the library (libpng) is not found,
#   shell commands 'action-if-not-found' is run. If 'action-if-not-found' is
#   not specified, the configuration exits on error, asking for a valid libpng
#   installation directory or --without-libpng.
#
#   If both header file and library are found, shell commands 'action-if-found'
#   is run. If 'action-if-found' is not specified, the default action sets
#   LIBPNG_CPPFLAGS to '-I${LIBPNG_HOME}/include', LIBPNG_LDFLAGS to
#   '-L${LIBPNG_HOME}/lib', and LIBPNG_LIBS to '-lpng', and calls
#   AC_DEFINE(HAVE_LIBPNG). You should use autoheader to include a definition
#   for this symbol in a config.h file. Sample usage in a C/C++ source
#   is as follows:
#
#     #ifdef HAVE_LIBPNG
#     #include <png.h>
#     #endif /* HAVE_LIBPNG */
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

AU_ALIAS([LIBPNG], [AX_LIBPNG])
AC_DEFUN([AX_LIBPNG],
#
# Handle user hints
#
[AC_MSG_CHECKING(if libpng is wanted)
libpng_places="/usr/local /usr /opt/local /sw"
AC_ARG_WITH([libpng],
[  --with-libpng=DIR         path to libpng installation @<:@defaults to
                          /usr/local or /usr if not found in /usr/local@:>@
  --without-libpng          disable libpng usage completely],
[if test "$withval" != no ; then
  with_libpng="$withval"
  AC_MSG_RESULT(yes)
  if test -d "$withval"
  then
    libpng_places="$withval $libpng_places"
  else
    AC_MSG_WARN([Sorry, $withval does not exist, checking usual places])
  fi
else
  with_libpng=no
  libpng_places=
  AC_MSG_RESULT(no)
fi],
[AC_MSG_RESULT(yes)])

#
# Locate libpng, if wanted
#
libpng_found=no
if test -n "${libpng_places}"
then
	# check the user supplied or any other more or less 'standard' place:
	#   Most UNIX systems      : /usr/local and /usr
	#   MacPorts / Fink on OSX : /opt/local respectively /sw
	for LIBPNG_HOME in ${libpng_places} ; do
	  if test -f "${LIBPNG_HOME}/include/png.h"; then break; fi
	  LIBPNG_HOME=""
	done

  LIBPNG_OLD_LDFLAGS=$LDFLAGS
  LIBPNG_OLD_CPPFLAGS=$CPPFLAGS
  if test -n "${LIBPNG_HOME}"; then
        LDFLAGS="$LDFLAGS -L${LIBPNG_HOME}/lib"
        CPPFLAGS="$CPPFLAGS -I${LIBPNG_HOME}/include"
  fi
  AC_LANG_PUSH([C])
  AC_CHECK_LIB([png], [png_get_io_ptr], [libpng_cv_libpng=yes], [libpng_cv_libpng=no])
  AC_CHECK_HEADER([png.h], [libpng_cv_png_h=yes], [libpng_cv_png_h=no])
  AC_LANG_POP([C])
  if test "$libpng_cv_libpng" = "yes" && test "$libpng_cv_png_h" = "yes"
  then
    #
    # If both library and header were found, action-if-found
    #
    libpng_found=yes
    m4_ifblank([$1],[
                LIBPNG_CPPFLAGS="-I${LIBPNG_HOME}/include"
                LIBPNG_LDFLAGS="-L${LIBPNG_HOME}/lib"
                LIBPNG_LIBS="-lpng"
                AC_DEFINE([HAVE_LIBPNG], [1],
                          [Define to 1 if you have libpng])
               ],[
                # Restore variables
                LDFLAGS="$LIBPNG_OLD_LDFLAGS"
                CPPFLAGS="$LIBPNG_OLD_CPPFLAGS"
                $1
               ])
  else
    #
    # If either header or library was not found, action-if-not-found
    #
    m4_default([$2],[
                AC_MSG_ERROR([either specify a valid libpng installation with --with-libpng=DIR or disable libpng usage with --without-libpng])
                ])
  fi
fi
])
