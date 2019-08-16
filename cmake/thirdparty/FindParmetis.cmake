# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

###############################################################################
#
# Setup Parmetis/Metis
# This file defines:
#  PARMETIS_FOUND - If Parmetis was found
#  PARMETIS_INCLUDE_DIR - The Parmetis include directories
#  PARMETIS_LIBRARY - The Parmetis library

# first Check for PARMETIS_DIR

if(NOT PARMETIS_DIR)
    MESSAGE(FATAL_ERROR "Could not find Metis. Parmetis support needs explicit PARMETIS_DIR")
endif()

find_path( 
    PARMETIS_INCLUDE_DIR parmetis.h
    PATHS  ${PARMETIS_DIR}/include/
    NO_DEFAULT_PATH           
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

find_library( 
    PARMETIS_LIBRARY NAMES parmetis libparmetis
    PATHS ${PARMETIS_DIR}/lib
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    PARMETIS
    DEFAULT_MSG
    PARMETIS_INCLUDE_DIR
    PARMETIS_LIBRARY
)

