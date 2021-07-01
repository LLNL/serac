# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option(ENABLE_ASAN "Enable AddressSanitizer for memory checking (Clang or GCC only)" OFF)
if(ENABLE_ASAN)
    if(NOT (C_COMPILER_FAMILY_IS_CLANG OR C_COMPILER_FAMILY_IS_GNU))
        message(FATAL_ERROR "ENABLE_ASAN only supports Clang and GCC")
    endif()
endif()

option(SERAC_ENABLE_LUMBERJACK "Enable Axom's Lumberjack component" ON)

# Only enable Serac's code checks by default if it is the top-level project
# or a user overrides it
if("${CMAKE_PROJECT_NAME}" STREQUAL "serac")
    set(_enable_serac_code_checks ON)
else()
    set(_enable_serac_code_checks OFF)
endif()
option(SERAC_ENABLE_CODE_CHECKS "Enable Serac's code checks" ${_enable_serac_code_checks})

#------------------------------------------------------------------------------
# Create symlink in installed bin
#------------------------------------------------------------------------------
if(GLVIS_EXECUTABLE)
    add_custom_target(glvis_symlink ALL
                      COMMAND ${CMAKE_COMMAND} 
                      -E create_symlink ${GLVIS_EXECUTABLE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/glvis)
endif()


#------------------------------------------------------------------------------
# Global includes (restrict these as much as possible)
#------------------------------------------------------------------------------
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})
