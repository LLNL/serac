# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(common_headers
    boundary_condition.hpp
    common.hpp
    serac_types.hpp
    logger.hpp
    mesh_utils.hpp
    terminator.hpp
    )

set(common_sources
    finite_element_state.cpp
    boundary_condition.cpp
    logger.cpp
    mesh_utils.cpp
    terminator.cpp
    )

blt_add_library(
    NAME        common
    HEADERS     ${common_headers}
    SOURCES     ${common_sources}
    DEPENDS_ON  axom mpi mfem
    )

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
