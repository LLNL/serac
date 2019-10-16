# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

###############################################################################
#
# BLT (and CMAKE) SERAC build system
#
###############################################################################

cmake_minimum_required(VERSION 3.3)

# Use the LLNL toss3 clang settings bundled with BLT
# include(cmake/defaults.cmake)

project(serac)

# Use c++11 standard everywhere
set(BLT_CXX_STD "c++11" CACHE STRING "")

###############################################################################
# MPI
###############################################################################
# MPI is required in serac. (unprotected usage in sources and tests)
# If ENABLE_MPI is not defined, we conclude no host-config is loaded and we
# want the default cmake behavior to be used (FindMPI).
if(NOT DEFINED ENABLE_MPI)
  set(ENABLE_MPI ON CACHE BOOL "")
  set(ENABLE_FINDMPI)
endif()

###############################################################################
# Setup BLT
###############################################################################

if(DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

###############################################################################
# Setup external libraries
###############################################################################

include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupSeracThirdParty.cmake)

###############################################################################
# Build src
###############################################################################

add_subdirectory(src)

###############################################################################
# Build tests
###############################################################################

if(ENABLE_TESTS AND ENABLE_GTEST)
    add_subdirectory(tests)
    endif()

