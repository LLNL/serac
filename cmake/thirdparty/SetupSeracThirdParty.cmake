# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (NOT SERAC_THIRD_PARTY_LIBRARIES_FOUND)
    # Prevent this file from being called twice in the same scope
    set(SERAC_THIRD_PARTY_LIBRARIES_FOUND TRUE)

    #------------------------------------------------------------------------------
    # CUDA
    #------------------------------------------------------------------------------
    if(ENABLE_CUDA)
        # Manually set includes as system includes
        foreach(_target cuda_runtime cuda)
            get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
            set_property(TARGET ${_target} 
                         APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                         "${_dirs}")
        endforeach()
    endif()

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

    set(_conduit_config "${CONDUIT_DIR}/lib/cmake/conduit/ConduitConfig.cmake")
    if(NOT EXISTS ${_conduit_config})
        MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file ${_conduit_config}")
    endif()

    find_package(Conduit REQUIRED
                 NO_DEFAULT_PATH
                 PATHS ${CONDUIT_DIR}/lib/cmake/conduit)

    # Manually set includes as system includes
    get_target_property(_dirs conduit::conduit INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET conduit::conduit 
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                 "${_dirs}")


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
    # Adiak
    #------------------------------------------------------------------------------
    if(ADIAK_DIR)
        serac_assert_is_directory(VARIABLE_NAME ADIAK_DIR)

        find_package(adiak REQUIRED NO_DEFAULT_PATH PATHS ${ADIAK_DIR})
        message(STATUS "Adiak support is ON")
        set(ADIAK_FOUND TRUE)
    else()
        message(STATUS "Adiak support is OFF")
        set(ADIAK_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # Caliper
    #------------------------------------------------------------------------------
    # TODO: temporarily turned off until PR # 626
    set(CALIPER_DIR "" CACHE STRING "" FORCE)
    if(CALIPER_DIR)
        serac_assert_is_directory(VARIABLE_NAME CALIPER_DIR)

        # Should this logic be in the Caliper CMake package?
        # If CMake version doesn't support CUDAToolkit the libraries
        # are just "baked in"
        if(ENABLE_CUDA)
            if(CMAKE_VERSION VERSION_LESS 3.17)
                message(FATAL_ERROR "Serac+Caliper+CUDA requires CMake > 3.17.")
            else()
                find_package(CUDAToolkit REQUIRED)
            endif() 
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
    # MFEM
    #------------------------------------------------------------------------------
    if(NOT SERAC_ENABLE_CODEVELOP)
        message(STATUS "Using installed MFEM")
        include(${CMAKE_CURRENT_LIST_DIR}/FindMFEM.cmake)
    else()
        message(STATUS "Using MFEM submodule")

        #### Store Data that MFEM clears
        set(tpls_to_save AMGX AXOM CALIPER CONDUIT HDF5
                         HYPRE LUA METIS NETCDF PETSC RAJA UMPIRE)
        foreach(_tpl ${tpls_to_save})
            set(${_tpl}_DIR_SAVE "${${_tpl}_DIR}")
        endforeach()

        #### MFEM "Use" Options

        # Assumes that we have AMGX if we have CUDA
        set(MFEM_USE_AMGX ${ENABLE_CUDA} CACHE BOOL "")
        set(MFEM_USE_CALIPER ${CALIPER_FOUND} CACHE BOOL "")
        # We don't use MFEM's Conduit/Axom support
        set(MFEM_USE_CONDUIT OFF CACHE BOOL "")
        set(MFEM_USE_CUDA ${ENABLE_CUDA} CACHE BOOL "")
        set(MFEM_USE_LAPACK ON CACHE BOOL "")
        # mfem+mpi requires metis
        set(MFEM_USE_METIS ${ENABLE_MPI} CACHE BOOL "")
        set(MFEM_USE_METIS_5 ${ENABLE_MPI} CACHE BOOL "")
        set(MFEM_USE_MPI ${ENABLE_MPI} CACHE BOOL "")
        if(NETCDF_DIR)
            serac_assert_is_directory(VARIABLE_NAME NETCDF_DIR)
            set(MFEM_USE_NETCDF ON CACHE BOOL "")
        endif()
        # mfem+mpi also needs parmetis
        if(ENABLE_MPI)
            serac_assert_is_directory(VARIABLE_NAME PARMETIS_DIR)
            # Slightly different naming convention
            set(ParMETIS_DIR ${PARMETIS_DIR} CACHE PATH "")
        endif()
        set(MFEM_USE_OPENMP ${ENABLE_OPENMP} CACHE BOOL "")
        set(MFEM_USE_PETSC ${PETSC_FOUND} CACHE BOOL "")
        set(MFEM_USE_RAJA ${RAJA_FOUND} CACHE BOOL "")
        if(SUNDIALS_DIR)
            serac_assert_is_directory(VARIABLE_NAME SUNDIALS_DIR)
            set(MFEM_USE_SUNDIALS ON CACHE BOOL "")
        else()
            set(MFEM_USE_SUNDIALS OFF CACHE BOOL "")
        endif()
        if(SUPERLUDIST_DIR)
            serac_assert_is_directory(VARIABLE_NAME SUPERLUDIST_DIR)
            # MFEM uses a slightly different naming convention
            set(SuperLUDist_DIR ${SUPERLUDIST_DIR} CACHE PATH "")
            set(MFEM_USE_SUPERLU ${ENABLE_MPI} CACHE BOOL "")
        endif()
        set(MFEM_USE_UMPIRE ${UMPIRE_FOUND} CACHE BOOL "")
        set(MFEM_USE_ZLIB ON CACHE BOOL "")

        #### MFEM Configuration Options

        # Prefix the "check" targets
        set(MFEM_CUSTOM_TARGET_PREFIX "mfem_" CACHE STRING "")

        # Tweaks needed after Spack converted to the HDF5 CMake build system
        set(HDF5_TARGET_NAMES "hdf5::hdf5-static;hdf5::hdf5-shared" CACHE STRING "")
        set(HDF5_IMPORT_CONFIG "RELWITHDEBINFO" CACHE STRING "")
        set(HDF5_C_LIBRARY_hdf5_hl "hdf5::hdf5_hl-static" CACHE STRING "")

        # Disable tests + examples
        set(MFEM_ENABLE_TESTING  OFF CACHE BOOL "")
        set(MFEM_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(MFEM_ENABLE_MINIAPPS OFF CACHE BOOL "")

        add_subdirectory(${PROJECT_SOURCE_DIR}/mfem  ${CMAKE_BINARY_DIR}/mfem)
 
        set(MFEM_FOUND TRUE CACHE BOOL "" FORCE)

        # Patch the mfem target with the correct include directories
        get_target_property(_mfem_includes mfem INCLUDE_DIRECTORIES)
        target_include_directories(mfem SYSTEM INTERFACE $<BUILD_INTERFACE:${_mfem_includes}>)
        target_include_directories(mfem SYSTEM INTERFACE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/mfem>)
        
        #### Restore previously stored data
        foreach(_tpl ${tpls_to_save})
            set(${_tpl}_DIR "${${_tpl}_DIR_SAVE}" CACHE PATH "" FORCE)
        endforeach()

        set(MFEM_BUILT_WITH_CMAKE TRUE)
    endif()

    #------------------------------------------------------------------------------
    # Axom
    #------------------------------------------------------------------------------
    if(NOT SERAC_ENABLE_CODEVELOP)
        message(STATUS "Using installed Axom")
        serac_assert_is_directory(VARIABLE_NAME AXOM_DIR)

        find_package(axom REQUIRED
                        NO_DEFAULT_PATH 
                        PATHS ${AXOM_DIR}/lib/cmake)

        message(STATUS "Axom support is ON")

        #
        # Check for optional Axom headers that are required for Serac
        #

        # sol.hpp
        find_path(
            _sol_found sol.hpp
            PATHS ${AXOM_DIR}/include/axom
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
        )
        if (NOT _sol_found)
            message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: axom/sol.hpp"
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
    else()
        # Otherwise we use the submodule
        message(STATUS "Using Axom submodule")
        if(NOT LUA_DIR)
            message(FATAL_ERROR "LUA_DIR is required to use the Axom submodule"
                                "\nTry running CMake with '-DLUA_DIR=path/to/lua/install'\n ")
        endif()
        set(AXOM_ENABLE_MFEM_SIDRE_DATACOLLECTION ON CACHE BOOL "")
        set(AXOM_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(AXOM_ENABLE_TESTS    OFF CACHE BOOL "")
        set(AXOM_ENABLE_DOCS     OFF CACHE BOOL "")
        set(AXOM_ENABLE_TOOLS    OFF CACHE BOOL "")
        set(AXOM_USE_CALIPER ${CALIPER_FOUND} CACHE BOOL "")

        # Used for the doxygen target
        set(AXOM_CUSTOM_TARGET_PREFIX "axom_" CACHE STRING "" FORCE)
        if(ENABLE_CUDA)
            # This appears to be unconditionally needed for Axom, why isn't it part of the build system?
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
        endif()
        add_subdirectory(${PROJECT_SOURCE_DIR}/axom/src ${CMAKE_BINARY_DIR}/axom)
        set(AXOM_FOUND TRUE CACHE BOOL "" FORCE)

        # Mark the axom includes as "system" and filter unallowed directories
        get_target_property(_dirs axom INTERFACE_INCLUDE_DIRECTORIES)
        list(REMOVE_ITEM _dirs ${PROJECT_SOURCE_DIR})
        set_property(TARGET axom 
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     "${_dirs}")
        set_property(TARGET axom 
                     APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                     "${_dirs}")

    endif()

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
    # Umpire
    #------------------------------------------------------------------------------
    if(UMPIRE_DIR)
        serac_assert_is_directory(VARIABLE_NAME UMPIRE_DIR)
        find_package(umpire REQUIRED NO_DEFAULT_PATH 
                     PATHS ${UMPIRE_DIR})
        message(STATUS "Umpire support is ON")
        set(UMPIRE_FOUND TRUE)
    else()
        message(STATUS "Umpire support is OFF")
        set(UMPIRE_FOUND FALSE)
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
