# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(problem_description_headers
    boundary_condition.hpp
    equation_solver.hpp
    finite_element_state.hpp
    )

set(problem_description_sources
    boundary_condition.cpp
    equation_solver.cpp
    finite_element_state.cpp
    )

set(problem_description_depends infrastructure mfem mpi axom)

blt_add_library(
    NAME        problem_description
    HEADERS     ${problem_description_headers}
    SOURCES     ${problem_description_sources}
    DEPENDS_ON  ${problem_description_depends}
    )


target_include_directories(problem_description PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
