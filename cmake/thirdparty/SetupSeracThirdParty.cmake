# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (NOT SERAC_THIRD_PARTY_LIBRARIES_FOUND)
    # Prevent this file from being called twice in the same scope
    set(SERAC_THIRD_PARTY_LIBRARIES_FOUND TRUE)

    # Policy to use <PackageName>_ROOT variable in find_<Package> commands
    # Policy added in 3.12+
    if(POLICY CMP0074)
        set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
    endif()

    #------------------------------------------------------------------------------
    # Conduit (required by Axom and Ascent)
    #------------------------------------------------------------------------------
    if(NOT CONDUIT_DIR)
        MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
    endif()

    if(NOT WIN32)
        set(_conduit_config "${CONDUIT_DIR}/lib/cmake/ConduitConfig.cmake")
        if(NOT EXISTS ${_conduit_config})
            MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file ${_conduit_config}")
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
    get_target_property(_dirs conduit::conduit INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET conduit::conduit 
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                 "${_dirs}")


    #------------------------------------------------------------------------------
    # Ascent
    #------------------------------------------------------------------------------
    if(NOT ASCENT_DIR)
        MESSAGE(FATAL_ERROR "Could not find Ascent. Ascent requires explicit ASCENT_DIR.")
    endif()

    if(NOT WIN32)
        set(_ascent_config "${ASCENT_DIR}/lib/cmake/ascent/AscentConfig.cmake")
        if(NOT EXISTS ${_ascent_config})
            MESSAGE(FATAL_ERROR "Could not find Ascent CMake include file ${_ascent_config}")
        endif()

        find_package(Ascent REQUIRED
                     NO_DEFAULT_PATH
                     PATHS ${ASCENT_DIR}/lib/cmake)
    else()
        # Allow for several different configurations of Ascent
        find_package(Ascent CONFIG 
            REQUIRED
            HINTS ${ASCENT_DIR}/cmake/ascent 
                  ${ASCENT_DIR}/lib/cmake/ascent
                  ${ASCENT_DIR}/share/cmake/ascent
                  ${ASCENT_DIR}/share/ascent
                  ${ASCENT_DIR}/cmake)
    endif()

    # Manually set includes as system includes
    get_target_property(_dirs ascent::ascent INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET ascent::ascent 
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                 "${_dirs}")


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

    #
    # Check for optional Axom headers that are required for Serac
    #

    # sol.hpp
    find_path(
        _sol_found sol.hpp
        PATHS ${AXOM_DIR}/include/sol
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )
    if (NOT _sol_found)
        message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: sol/sol.hpp"
                            "\nTry building Axom with '-DBLT_CXX_STD=c++14' or higher\n ")
    endif()

    # LuaReader.hpp
    find_path(
        _luareader_found LuaReader.hpp
        PATHS ${AXOM_DIR}/include/axom/inlet
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )
    if (NOT _luareader_found)
        message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: axom/inlet/LuaReader.hpp"
                            "\nTry building Axom with '-DLUA_DIR=path/to/lua/install'\n ")
    endif()

    # MFEMSidreDataCollection.hpp
    find_path(
        _mfemdatacollection_found MFEMSidreDataCollection.hpp
        PATHS ${AXOM_DIR}/include/axom/sidre/core
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )
    if (NOT _mfemdatacollection_found)
        message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: axom/sidre/core/MFEMSidreDataCollection.hpp"
                            "\nTry building Axom with '-DAXOM_ENABLE_MFEM_SIDRE_DATACOLLECTION=ON'\n ")
    endif()


    #------------------------------------------------------------------------------
    # MFEM
    #------------------------------------------------------------------------------
    include(${CMAKE_CURRENT_LIST_DIR}/FindMFEM.cmake)


    #------------------------------------------------------------------------------
    # Tribol
    #------------------------------------------------------------------------------
    if(TRIBOL_DIR)
        serac_assert_is_directory(VARIABLE_NAME TRIBOL_DIR)

        find_package(tribol REQUIRED
                            NO_DEFAULT_PATH 
                            PATHS ${TRIBOL_DIR}/lib/cmake)

        if(TARGET tribol)
            message(STATUS "Tribol CMake exported library loaded: tribol")
        else()
            message(FATAL_ERROR "Could not load Tribol CMake exported library: tribol")
        endif()

        # Set include dir to system
        set(TRIBOL_INCLUDE_DIR ${TRIBOL_DIR}/include)
        set_property(TARGET tribol
                     APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                     ${TRIBOL_INCLUDE_DIR})
    else()
        set(TRIBOL_FOUND OFF)
    endif()
    
    message(STATUS "Tribol support is " ${TRIBOL_FOUND})


    #------------------------------------------------------------------------------
    # Caliper
    #------------------------------------------------------------------------------
    if(CALIPER_DIR)
        serac_assert_is_directory(VARIABLE_NAME CALIPER_DIR)

        # Should this logic be in the Caliper CMake package?
        # If CMake version doesn't support CUDAToolkit the libraries
        # are just "baked in"
        if(ENABLE_CUDA AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
            find_package(CUDAToolkit REQUIRED)
        endif()

        find_package(caliper REQUIRED NO_DEFAULT_PATH 
                     PATHS ${CALIPER_DIR})
        message(STATUS "Caliper support is ON")
        set(CALIPER_FOUND TRUE)

        # Set the include directories as Caliper does not completely
        # configure the "caliper" target
        set_target_properties(caliper PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${caliper_INCLUDE_PATH})
    else()
        message(STATUS "Caliper support is OFF")
        set(CALIPER_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # PETSC
    #------------------------------------------------------------------------------
    if(PETSC_DIR)
        serac_assert_is_directory(VARIABLE_NAME PETSC_DIR)
        include(${CMAKE_CURRENT_LIST_DIR}/FindPETSc.cmake)
        message(STATUS "PETSc support is ON")
        set(PETSC_FOUND TRUE)
    else()
        message(STATUS "PETSc support is OFF")
        set(PETSC_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # Remove exported OpenMP flags because they are not language agnostic
    #------------------------------------------------------------------------------
    set(_props)
    if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
        list(APPEND _props INTERFACE_LINK_OPTIONS)
    endif()
    list(APPEND _props INTERFACE_COMPILE_OPTIONS)

    # This flag is empty due to us not enabling fortran but we need to strip it
    # so it doesn't propagate in our project
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

    #---------------------------------------------------------------------------
    # Remove non-existant INTERFACE_INCLUDE_DIRECTORIES from imported targets
    # to work around CMake error
    #---------------------------------------------------------------------------
    set(_imported_targets
        ascent::ascent
        axom
        conduit
        conduit::conduit_mpi
        conduit::conduit
        conduit_relay_mpi
        conduit_relay_mpi_io
        conduit_blueprint
        conduit_blueprint_mpi)

    foreach(_target ${_imported_targets})
        if(TARGET ${_target})
            message(STATUS "Removing non-existant include directories from target[${_target}]")

            get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
            set(_existing_dirs)
            foreach(_dir ${_dirs})
                if (EXISTS "${_dir}")
                    list(APPEND _existing_dirs "${_dir}")
                endif()
            endforeach()
            if (_existing_dirs)
                set_target_properties(${_target} PROPERTIES
                                      INTERFACE_INCLUDE_DIRECTORIES "${_existing_dirs}" )
            endif()
        endif()
    endforeach()

    # List of TPL targets built in to BLT - will need to be adjusted when we start using HIP
    set(TPL_DEPS)
    blt_list_append(TO TPL_DEPS ELEMENTS cuda cuda_runtime IF ENABLE_CUDA)
    blt_list_append(TO TPL_DEPS ELEMENTS mpi IF ENABLE_MPI)

    foreach(dep ${TPL_DEPS})
        # If the target is EXPORTABLE, add it to the export set
        get_target_property(_is_imported ${dep} IMPORTED)
        if(NOT ${_is_imported})
            install(TARGETS              ${dep}
                    EXPORT               serac-targets
                    DESTINATION          lib)
        endif()
    endforeach()
endif()
