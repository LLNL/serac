# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

###############################################################################
#
# Setup HYPRE
# This file defines:
#  HYPRE_FOUND - If HYPRE was found
#  HYPRE_INCLUDE_DIRS - The HYPRE include directories
#  HYPRE_LIBRARY - The HYPRE library

# first Check for HYPRE_DIR
#                 HYPRE_INCSUBDIR
#                 HYPRE_LIBSUBDIR

if(NOT HYPRE_DIR)
    MESSAGE(FATAL_ERROR "Could not find HYPRE. HYPRE support needs explicit HYPRE_DIR")
endif()

#find includes
find_path( 
    HYPRE_INCLUDE_DIRS HYPRE.h
    PATHS  ${HYPRE_DIR}/include
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

find_library( 
    HYPRE_LIBRARY 
    NAMES HYPRE libHYPRE
    PATHS ${HYPRE_DIR}/lib
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    HYPRE
    DEFAULT_MSG
    HYPRE_INCLUDE_DIRS
    HYPRE_LIBRARY
)
