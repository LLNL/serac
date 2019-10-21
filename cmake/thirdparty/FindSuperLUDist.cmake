# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

###############################################################################
#
# Setup SUPERLUDIST
# This file defines:
#  SUPERLUDIST_FOUND - If SUPERLUDIST was found
#  SUPERLUDIST_INCLUDE_DIRS - The SUPERLUDIST include directories
#  SUPERLUDIST_LIBRARY - The SUPERLUDIST library

# first Check for SUPERLUDIST_DIR
#                 SUPERLUDIST_INCSUBDIR
#                 SUPERLUDIST_LIBSUBDIR

if(NOT SUPERLUDIST_DIR)
    MESSAGE(FATAL_ERROR "Could not find SUPERLUDIST. SUPERLUDIST support needs explicit SUPERLUDIST_DIR")
endif()

#find includes
find_path( SUPERLUDIST_INCLUDE_DIRS superlu_ddefs.h
           PATHS  ${SUPERLUDIST_DIR}/SRC
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( SUPERLUDIST_LIBRARY NAMES superlu_dist_4.2
              PATHS ${SUPERLUDIST_DIR}/lib
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)


include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SUPERLUDIST  DEFAULT_MSG
                                  SUPERLUDIST_INCLUDE_DIRS
                                  SUPERLUDIST_LIBRARY )
