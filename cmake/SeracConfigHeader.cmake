# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Version information that go into the generated config header
#------------------------------------------------------------------------------
set(SERAC_VERSION_MAJOR 0)
set(SERAC_VERSION_MINOR 0)
set(SERAC_VERSION_PATCH 1)
string(CONCAT SERAC_VERSION_FULL
    "v${SERAC_VERSION_MAJOR}"
    ".${SERAC_VERSION_MINOR}"
    ".${SERAC_VERSION_PATCH}" )

if (Git_FOUND)
  ## check to see if we are building from a Git repo or an exported tarball
  blt_is_git_repo( OUTPUT_STATE is_git_repo )

  if(${is_git_repo})
    blt_git_hashcode(HASHCODE sha1 RETURN_CODE rc)
    if(NOT ${rc} EQUAL 0)
      message(FATAL_ERROR "blt_git_hashcode failed!")
    endif()

    set(SERAC_GIT_SHA ${sha1})
  endif()

endif()

message(STATUS "Configuring Serac version ${SERAC_VERSION_FULL}")


#------------------------------------------------------------------------------
# Create variable for every TPL
#------------------------------------------------------------------------------
set(TPL_DEPS AXOM CONDUIT FMT HDF5 MFEM MPI TRIBOL )
foreach(dep ${TPL_DEPS})
    if( ${dep}_FOUND OR ENABLE_${dep} )
        set(SERAC_USE_${dep} TRUE)
    endif()
endforeach()


#--------------------------------------------------------------------------
# Add define we can use when debug builds are enabled
#--------------------------------------------------------------------------
set(SERAC_DEBUG FALSE)
if(CMAKE_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)")
    set(SERAC_DEBUG TRUE)
endif()


#------------------------------------------------------------------------------
# General Build Info
#------------------------------------------------------------------------------
serac_convert_to_native_escaped_file_path(${PROJECT_SOURCE_DIR} SERAC_SRC_DIR)
serac_convert_to_native_escaped_file_path(${CMAKE_BINARY_DIR}   SERAC_BIN_DIR)


#------------------------------------------------------------------------------
# Create Config Header
#------------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/src/serac_config.hpp.in
    ${CMAKE_BINARY_DIR}/include/serac_config.hpp
)

install(FILES ${CMAKE_BINARY_DIR}/include/serac_config.hpp DESTINATION include)
