# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)


#------------------------------------------------------------------------------
# Axom
#------------------------------------------------------------------------------
if(AXOM_DIR)
    serac_assert_is_directory(VARIABLE_NAME AXOM_DIR)

    set(AXOM_INCLUDE_DIR ${AXOM_DIR}/include)

    set(AXOM_LIBRARIES fmt sparsehash axom )
    foreach(_library ${AXOM_LIBRARIES})
        set(_target_file ${AXOM_DIR}/lib/cmake/${_library}-targets.cmake)

        if(NOT EXISTS ${_target_file})
            MESSAGE(FATAL_ERROR "Could not find Axom CMake exported target file (${_target_file})")
        endif()

        include(${_target_file})

        # Set include dir to system
        set_property(TARGET ${_library}
                    APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                    ${AXOM_INCLUDE_DIR})
    endforeach()

    # Sets AXOM_FOUND if AXOM_INCLUDE_DIRS and AXOM_LIBRARIES are not empty
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(AXOM  DEFAULT_MSG
                                      AXOM_INCLUDE_DIRS
                                      AXOM_LIBRARIES )
else()
    message(STATUS "Axom support is OFF")
    set(AXOM_FOUND FALSE CACHE BOOL "")
endif()


#------------------------------------------------------------------------------
# MFEM
#------------------------------------------------------------------------------
include(cmake/thirdparty/FindMFEM.cmake)


#------------------------------------------------------------------------------
# TRIBOL
#------------------------------------------------------------------------------
if (TRIBOL_DIR)
    serac_assert_is_directory(VARIABLE_NAME TRIBOL_DIR)

    find_package(tribol REQUIRED PATHS ${TRIBOL_DIR})

    message(STATUS "Checking for expected TRIBOL target 'tribol'")
    if (NOT TARGET tribol)
        message(FATAL_ERROR "TRIBOL failed to load: ${TRIBOL_DIR}")
    else()
        message(STATUS "TRIBOL loaded: ${TRIBOL_DIR}")
        set_property(TARGET tribol 
                     APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                     ${TRIBOL_INCLUDE_DIRS})
        set(TRIBOL_FOUND TRUE CACHE BOOL "")
    endif()
else()
    message(STATUS "TRIBOL support is OFF")
    set(TRIBOL_FOUND FALSE CACHE BOOL "")
endif()
