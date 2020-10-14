# The MIT License (MIT)

# Copyright (c) 2016-2019 Simon Praetorius
#               2019      Felix Müller

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# FindPETSc.cmake
#
# Finds the PETSc library
#
# This will define the following variables
#
#    PETSc_FOUND
#    PETSc_VERSION
#
# and the following imported targets
#
#    PETSc::PETSc
#
# Author: Simon Praetorius <simon.praetorius@tu-dresden.de>

include(FindPkgConfig)

if (NOT PKG_CONFIG_FOUND)
  message(FATAL_ERROR "Can not find PkgConfig!")
endif()

mark_as_advanced(PETSc_FOUND PETSc_VERSION PETSC_PKG_CONFIG)

find_path(PETSC_PKG_CONFIG "PETSc.pc"
  HINTS
    ${PETSC_DIR}
    ${PETSC_ROOT}
    ENV PETSC_DIR
    ENV PETSC_ROOT
    ENV PKG_CONFIG_PATH
  PATHS
    /etc/alternatives
    /usr/lib/petsc
    /usr/lib/petsc/linux-gnu-cxx-opt
    /usr/lib/petsc/linux-gnu-c-opt
  PATH_SUFFIXES lib/pkgconfig/
)

if (PETSC_PKG_CONFIG)
  set(ENV{PKG_CONFIG_PATH} "${PETSC_PKG_CONFIG}:$ENV{PKG_CONFIG_PATH}")
endif (PETSC_PKG_CONFIG)

if (PETSc_FIND_VERSION)
  pkg_check_modules(PETSC PETSc>=${PETSc_FIND_VERSION})
else ()
  pkg_check_modules(PETSC PETSc)
endif ()

if (PETSC_STATIC_FOUND)
  set(_prefix PETSC_STATIC)
elseif (PETSC_FOUND)
  set(_prefix PETSC)
endif ()

set(PETSc_VERSION "${${_prefix}_VERSION}")
if ((PETSC_STATIC_FOUND OR PETSC_FOUND) AND NOT TARGET PETSc::PETSc)
  add_library(PETSc::PETSc INTERFACE IMPORTED GLOBAL)
  if (${_prefix}_INCLUDE_DIRS)
    set_property(TARGET PETSc::PETSc PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${${_prefix}_INCLUDE_DIRS}")
  endif ()
  if (${_prefix}_LINK_LIBRARIES)
    set_property(TARGET PETSc::PETSc PROPERTY
                 INTERFACE_LINK_LIBRARIES "${${_prefix}_LINK_LIBRARIES}")
  else ()
    # extract the absolute paths of link libraries from the LDFLAGS
    include(PkgConfigLinkLibraries)
    pkg_config_link_libraries(${_prefix} _libs)
    set_property(TARGET PETSc::PETSc PROPERTY
                 INTERFACE_LINK_LIBRARIES "${_libs}")
    unset(_libs)
  endif ()
  if (${_prefix}_LDFLAGS_OTHER)
    set_property(TARGET PETSc::PETSc PROPERTY
                 INTERFACE_LINK_OPTIONS "${${_prefix}_LDFLAGS_OTHER}")
  endif ()
  if (${_prefix}_CFLAGS_OTHER)
    set_property(TARGET PETSc::PETSc PROPERTY
                 INTERFACE_COMPILE_OPTIONS "${${_prefix}_CFLAGS_OTHER}")
  endif ()
  # workaround for PETSc macros redefining MPI functions
  set_property(TARGET PETSc::PETSc PROPERTY
               INTERFACE_COMPILE_DEFINITIONS "PETSC_HAVE_BROKEN_RECURSIVE_MACRO=1")
endif ()
unset(_prefix)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
  REQUIRED_VARS PETSc_VERSION
  VERSION_VAR PETSc_VERSION
)

# SERAC EDIT BEGIN

# PETSc variables
if(EXISTS ${PETSC_DIR}/conf/petscvariables)
  file(STRINGS ${PETSC_DIR}/conf/petscvariables
    PETSC_VARIABLES NEWLINE_CONSUME)
elseif(EXISTS ${PETSC_DIR}/lib/petsc/conf/petscvariables)
  file(STRINGS ${PETSC_DIR}/lib/petsc/conf/petscvariables
    PETSC_VARIABLES NEWLINE_CONSUME)
else()
  message(SEND_ERROR "PETSc variables not found")
endif()

# PETSC dependency resolution
string(REGEX MATCH "PETSC_EXTERNAL_LIB_BASIC = [^\n\r]*" PETSC_EXTERNAL_LIB_BASIC ${PETSC_VARIABLES})
string(REPLACE "PETSC_EXTERNAL_LIB_BASIC = " "" PETSC_EXTERNAL_LIB_BASIC ${PETSC_EXTERNAL_LIB_BASIC})
string(STRIP ${PETSC_EXTERNAL_LIB_BASIC} PETSC_EXTERNAL_LIB_BASIC)

string(REGEX MATCH "PETSC_CC_INCLUDES = [^\n\r]*" PETSC_CC_INCLUDES ${PETSC_VARIABLES})
string(REPLACE "PETSC_CC_INCLUDES = " "" PETSC_CC_INCLUDES ${PETSC_CC_INCLUDES})
string(STRIP ${PETSC_CC_INCLUDES} PETSC_CC_INCLUDES)

get_target_property(PETSC_INCLUDE_DIRS PETSc::PETSc INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(PETSC_LIB PETSc::PETSc INTERFACE_LINK_LIBRARIES)

blt_register_library(
    NAME          PETSc
    INCLUDES      ${PETSC_INCLUDE_DIRS} ${PETSC_CC_INCLUDES}
    LIBRARIES     ${PETSC_LIB} ${PETSC_EXTERNAL_LIB_BASIC}
    TREAT_INCLUDES_AS_SYSTEM ON)

# SERAC EDIT END
