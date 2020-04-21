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
            message(FATAL_ERROR "Could not find Axom CMake exported target file (${_target_file})")
        endif()

        include(${_target_file})

        if(TARGET ${_library})
            message(STATUS "Axom CMake exported library loaded: ${_library}")
        else()
            message(FATAL_ERROR "Could not load Axom CMake exported library: ${_library}")
        endif()

        # Set include dir to system
        set_property(TARGET ${_library}
                    APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                    ${AXOM_INCLUDE_DIR})

        string(TOUPPER ${_library} _ucname)
        set(${_ucname}_FOUND TRUE CACHE BOOL "")
    endforeach()
else()
    message(STATUS "Axom support is OFF")
    set(AXOM_FOUND FALSE CACHE BOOL "")
endif()


#------------------------------------------------------------------------------
# Conduit (required by Axom)
#------------------------------------------------------------------------------
if (AXOM_FOUND)
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
else()
    set(CONDUIT_FOUND FALSE CACHE BOOL "")
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

# Remove exported OpenMP flags because they are not language agnostic
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

