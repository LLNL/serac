# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)


# Policy to use <PackageName>_ROOT variable in find_<Package> commands
# Policy added in 3.12+
if(POLICY CMP0074)
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
endif()

#------------------------------------------------------------------------------
# Conduit (required by Axom)
#------------------------------------------------------------------------------
if(NOT CONDUIT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
endif()

if(NOT WIN32)
    set(_conduit_config "${CONDUIT_DIR}/lib/cmake/ConduitConfig.cmake")
    if(NOT EXISTS ${_conduit_config})
        MESSAGE(FATAL_ERROR "Could not find Conduit cmake include file ${_conduit_config}")
    endif()

    find_package(Conduit REQUIRED
                 NO_DEFAULT_PATH
                 PATHS ${CONDUIT_DIR}/lib/cmake)
else()
    # Allow for several different configurations of Conduit
    find_package(Conduit CONFIG 
        REQUIRED
        HINTS ${CONDUIT_DIR}/cmake/conduit 
              ${CONDUIT_DIR}/lib/cmake/conduit
              ${CONDUIT_DIR}/share/cmake/conduit
              ${CONDUIT_DIR}/share/conduit
              ${CONDUIT_DIR}/cmake)
endif()

# Manually set includes as system includes
set_property(TARGET conduit::conduit 
             APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
             "${CONDUIT_INSTALL_PREFIX}/include/")

set_property(TARGET conduit::conduit 
             APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
             "${CONDUIT_INSTALL_PREFIX}/include/conduit/")


#------------------------------------------------------------------------------
# Axom
#------------------------------------------------------------------------------
if(NOT AXOM_DIR)
  MESSAGE(FATAL_ERROR "Could not find Axom. Axom requires explicit AXOM_DIR.")
endif()

serac_assert_is_directory(VARIABLE_NAME AXOM_DIR)

find_package(axom REQUIRED
                  NO_DEFAULT_PATH 
                  PATHS ${AXOM_DIR}/lib/cmake)


#------------------------------------------------------------------------------
# MFEM
#------------------------------------------------------------------------------
include(cmake/thirdparty/FindMFEM.cmake)


#------------------------------------------------------------------------------
# Tribol
#------------------------------------------------------------------------------
if(TRIBOL_DIR)
    serac_assert_is_directory(VARIABLE_NAME TRIBOL_DIR)

    set(TRIBOL_INCLUDE_DIR ${TRIBOL_DIR}/include)

    set(_target_file ${TRIBOL_DIR}/lib/cmake/tribol-targets.cmake)

    if(NOT EXISTS ${_target_file})
        message(FATAL_ERROR "Could not find Tribol CMake exported target file (${_target_file})")
    endif()

    include(${_target_file})

    if(TARGET tribol)
        message(STATUS "Tribol CMake exported library loaded: tribol")
    else()
        message(FATAL_ERROR "Could not load Tribol CMake exported library: tribol")
    endif()

    # Set include dir to system
    set_property(TARGET tribol
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                 ${TRIBOL_INCLUDE_DIR})

    set(TRIBOL_FOUND TRUE CACHE BOOL "")
else()
    message(STATUS "Tribol support is OFF")
    set(TRIBOL_FOUND FALSE CACHE BOOL "")
endif()


#------------------------------------------------------------------------------
# Remove exported OpenMP flags because they are not language agnostic
#------------------------------------------------------------------------------
set(_props)
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
    list(APPEND _props INTERFACE_LINK_OPTIONS)
endif()
list(APPEND _props INTERFACE_COMPILE_OPTIONS)

# This flag is empty due to us not enabling fortran but we need to strip it so it doesnt propagate
# in our project
if("${OpenMP_Fortran_FLAGS}" STREQUAL "")
    set(OpenMP_Fortran_FLAGS "$<$<NOT:$<COMPILE_LANGUAGE:Fortran>>:-fopenmp=libomp>;$<$<COMPILE_LANGUAGE:Fortran>:-fopenmp>")
endif()

foreach(_target axom)
    if(TARGET ${_target})
        message(STATUS "Removing OpenMP Flags from target[${_target}]")

        foreach(_prop ${_props})
            get_target_property(_flags ${_target} ${_prop})
            if ( _flags )
                string( REPLACE "${OpenMP_CXX_FLAGS}" ""
                        correct_flags "${_flags}" )
                string( REPLACE "${OpenMP_Fortran_FLAGS}" ""
                        correct_flags "${correct_flags}" )

                set_target_properties( ${_target} PROPERTIES ${_prop} "${correct_flags}" )
            endif()
        endforeach()
    endif()
endforeach()

#------------------------------------------------------------------------------
# Check for Lua
#------------------------------------------------------------------------------
  find_path(
  LUA_INCLUDE_DIRS lua.hpp
  PATHS ${LUA_DIR}/include
  NO_DEFAULT_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_CMAKE_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_SYSTEM_PATH
  )

find_library(
  LUA_LIBRARIES NAMES lua
  PATHS ${LUA_DIR}/lib
  NO_DEFAULT_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_CMAKE_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_SYSTEM_PATH )

message(STATUS "lua libraries ${LUA_LIBRARIES}]")

blt_register_library(
    NAME          lua
    INCLUDES      ${LUA_INCLUDE_DIRS}
    LIBRARIES     ${LUA_LIBRARIES}
    TREAT_INCLUDES_AS_SYSTEM ON)

