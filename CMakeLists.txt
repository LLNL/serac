# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Basic CMake Setup
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)

project(serac LANGUAGES CXX)

# MPI is required in Serac.
set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Setup BLT
#------------------------------------------------------------------------------

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

# Tune BLT to our needs
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++17" CACHE STRING "")
endif()
set(ENABLE_ASTYLE      OFF CACHE BOOL "")
set(ENABLE_CLANGQUERY  OFF CACHE BOOL "")
set(ENABLE_UNCRUSTIFY  OFF CACHE BOOL "")

if (ENABLE_CUDA AND ON_BLUEOS)
    # Very specific fix for working around CMake adding implicit link directories returned by the BlueOS
    # compilers to link CUDA executables 
    set(BLT_CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#------------------------------------------------------------------------------
# Setup Macros and dependencies
#------------------------------------------------------------------------------

include(${PROJECT_SOURCE_DIR}/cmake/SeracMacros.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/SeracBasics.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/SeracCompilerFlags.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupSeracThirdParty.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/SeracConfigHeader.cmake)

#------------------------------------------------------------------------------
# Build src
#------------------------------------------------------------------------------

add_subdirectory(src)

#------------------------------------------------------------------------------
# Build tests
#------------------------------------------------------------------------------

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

serac_add_code_checks(PREFIX   serac
                      INCLUDES /src/ /tests/
                      EXCLUDES cmake/blt uberenv_libs)
