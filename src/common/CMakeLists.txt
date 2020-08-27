# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(common_headers
    boundary_condition.hpp
    common.hpp
    finite_element_state.hpp
    initialize.hpp
    serac_types.hpp
    logger.hpp
    mesh_utils.hpp
    profiling.hpp
    terminator.hpp
    )

set(common_sources
    finite_element_state.cpp
    boundary_condition.cpp
    initialize.cpp
    logger.cpp
    mesh_utils.cpp
    profiling.cpp
    terminator.cpp
    )

set(common_depends axom mpi mfem)
blt_list_append( TO common_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

blt_add_library(
    NAME        common
    HEADERS     ${common_headers}
    SOURCES     ${common_sources}
    DEPENDS_ON  ${common_depends}
    )


target_include_directories(common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${common_headers} DESTINATION include/common )
