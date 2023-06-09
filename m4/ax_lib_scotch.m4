# ===========================================================================
#       https://www.gnu.org/software/autoconf-archive/ax_lib_scotch.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_LIB_SCOTCH([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
#
# DESCRIPTION
#
#   This macro searches for the SCOTCH library in the user specified
#   location. The user may specify the location either by defining the
#   environment variable SCOTCH or by using the --with-scotch option to
#   configure. If the environment variable is defined it has precedent over
#   everything else. If no location was specified then it searches in
#   /usr/lib and /usr/local/lib for the library and in /usr/include and
#   /usr/local/include for the header files. Upon successful completion the
#   variables SCOTCH_LIB and SCOTCH_INCLUDE are set.
#
#   ACTION-IF-FOUND is a list of shell commands to run if a SCOTCH library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it is
#   not found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_SCOTCH. If ACTION-IF-NOT-FOUND is not specified then an error
#   will be generated halting configure.
#
# LICENSE
#
#   Copyright (c) 2008 Ben Bergen <ben@cs.fau.de>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.

#serial 11

AU_ALIAS([IMMDX_LIB_SCOTCH], [AX_LIB_SCOTCH])
AC_DEFUN([AX_LIB_SCOTCH], [
	AC_MSG_CHECKING(for SCOTCH library)
	AC_REQUIRE([AC_PROG_CC])
	#
	# User hints...
	#
	AC_ARG_VAR([SCOTCH], [SCOTCH library location])
	AC_ARG_WITH([scotch],
		[AS_HELP_STRING([--with-scotch],
		[user defined path to SCOTCH library])],
		[
			if test -n "$SCOTCH" ; then
				AC_MSG_RESULT(yes)
				with_scotch=$SCOTCH
			elif test "$withval" != no ; then
				AC_MSG_RESULT(yes)
				with_scotch=$withval
			else
				AC_MSG_RESULT(no)
			fi
		],
		[
			if test -n "$SCOTCH" ; then
				with_scotch=$SCOTCH
				AC_MSG_RESULT(yes)
			else
				with_scotch=/usr
				if test ! -f "$with_scotch/include/scotch.h" ; then
					with_scotch=/usr/local
					if test ! -f "$with_scotch/include/scotch.h" ; then
						with_scotch=""
						AC_MSG_RESULT(failed)
					else
						AC_MSG_RESULT(yes)
					fi
				else
					AC_MSG_RESULT(yes)
				fi
			fi
		])
	#
	# locate SCOTCH library
	#
		if test -n "$with_scotch" ; then
			old_CFLAGS=$CFLAGS
			old_LDFLAGS=$LDFLAGS
			CFLAGS="-I$with_scotch/include"
			LDFLAGS="-L$with_scotch/lib"

			AC_LANG_PUSH([C])

			AC_CHECK_LIB(scotch, SCOTCH_graphInit,
				[scotch_lib=yes], [scotch_lib=yes], [-lm])
			AC_CHECK_HEADER(scotch.h, [scotch_h=yes],
				[scotch_h=no])

			AC_LANG_POP([C])

			CFLAGS=$old_CFLAGS
			LDFLAGS=$old_LDFLAGS

			AC_MSG_CHECKING(SCOTCH in $with_scotch)
			if test "$scotch_lib" = "yes" -a "$scotch_h" = "yes" ; then
				AC_SUBST(SCOTCH_INCLUDE, [-I$with_scotch/include])
				AC_SUBST(SCOTCH_LIB, [-L$with_scotch/lib])
				AC_MSG_RESULT(ok)
			else
				AC_MSG_RESULT(failed)
			fi
		fi
		#
		#
		#
		if test x = x"$SCOTCH_LIB" ; then
			ifelse([$2],,[AC_MSG_ERROR(Failed to find valid SCOTCH library)],[$2])
			:
		else
			ifelse([$1],,[AC_DEFINE(HAVE_SCOTCH,1,[Define if you have SCOTCH library])],[$1])
			:
		fi
	])dnl AX_LIB_SCOTCH
