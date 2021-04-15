# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Basic CMake Setup
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)

project(serac LANGUAGES CXX C)

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
set(ENABLE_BENCHMARKS  OFF CACHE BOOL "")

set(ENABLE_ALL_WARNINGS ON CACHE BOOL "")
set(ENABLE_WARNINGS_AS_ERRORS ON CACHE BOOL "")

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
# Configure examples
#------------------------------------------------------------------------------

add_subdirectory(examples)

#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

serac_add_code_checks(PREFIX   serac
                      INCLUDES /src/ /tests/
                      EXCLUDES cmake/blt uberenv_libs)

#------------------------------------------------------------------------------
# Export Targets
#------------------------------------------------------------------------------
set(exported_targets
    serac_coefficients
    serac_physics
    serac_infrastructure
    serac_numerics
    serac_physics_utilities
    serac_physics_operators
    serac_physics_integrators)

add_library(serac INTERFACE)
target_link_libraries(serac INTERFACE ${exported_targets})
install(TARGETS              serac
        EXPORT               serac-targets
        DESTINATION          lib
        )

install(EXPORT serac-targets 
        NAMESPACE serac::
        DESTINATION lib/cmake
        )
