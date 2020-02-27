# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Basic CMake Setup
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.9)

project(serac LANGUAGES CXX)

# MPI is required in Serac.
set(ENABLE_MPI ON CACHE BOOL "")

# Find devtools before loading BLT which sets up their targets
include(${PROJECT_SOURCE_DIR}/cmake/SeracFindDevtools.cmake)

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
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()
set(ENABLE_CLANGQUERY OFF CACHE BOOL "")
set(ENABLE_UNCRUSTIFY OFF CACHE BOOL "")

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#------------------------------------------------------------------------------
# Setup Macros and dependencies
#------------------------------------------------------------------------------

include(${PROJECT_SOURCE_DIR}/cmake/SeracMacros.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupSeracThirdParty.cmake)

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
# Copy utility TPL executables
#------------------------------------------------------------------------------

if(GLVIS_EXECUTABLE)
    ADD_CUSTOM_TARGET(link_target ALL
                      COMMAND ${CMAKE_COMMAND} 
                      -E create_symlink ${GLVIS_EXECUTABLE} ${CMAKE_INSTALL_BINDIR}/glvis)
endif()


#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

serac_add_code_checks(PREFIX   serac
                      INCLUDES src tests
                      EXCLUDES cmake/blt)
