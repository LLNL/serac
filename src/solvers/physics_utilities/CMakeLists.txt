# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(physics_utilities_headers
    boundary_condition.hpp
    equation_solver.hpp
    finite_element_state.hpp
    )

set(physics_utilities_sources
    boundary_condition.cpp
    equation_solver.cpp
    finite_element_state.cpp
    )

set(physics_utilities_depends infrastructure mfem mpi axom)

blt_add_library(
    NAME        physics_utilities
    HEADERS     ${physics_utilities_headers}
    SOURCES     ${physics_utilities_sources}
    DEPENDS_ON  ${physics_utilities_depends}
    )


target_include_directories(physics_utilities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
