# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(variational_form)

set(physics_utilities_headers
    boundary_condition.hpp
    boundary_condition_manager.hpp
    equation_solver.hpp
    finite_element_state.hpp
    solver_config.hpp
    )

set(physics_utilities_sources
    boundary_condition.cpp
    boundary_condition_manager.cpp
    equation_solver.cpp
    finite_element_state.cpp
    )

set(physics_utilities_depends serac_infrastructure)

blt_add_library(
    NAME        serac_physics_utilities
    HEADERS     ${physics_utilities_headers}
    SOURCES     ${physics_utilities_sources}
    DEPENDS_ON  ${physics_utilities_depends}
    )

install(FILES ${physics_utilities_headers} DESTINATION include/serac/physics/utilities )

install(TARGETS              serac_physics_utilities
        EXPORT               serac-targets
        DESTINATION          lib
        )
