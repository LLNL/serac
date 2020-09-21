# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(utilities)
add_subdirectory(operators)

set(physics_sources
    base_solver.cpp
    nonlinear_solid_solver.cpp
    thermal_solver.cpp
    thermal_solver_rework.cpp
    thermal_structural_solver.cpp
    elasticity_solver.cpp
    )

set(physics_headers
    base_solver.hpp
    nonlinear_solid_solver.hpp
    thermal_solver.hpp
    thermal_solver_rework.hpp
    thermal_structural_solver.hpp
    elasticity_solver.hpp
    )

blt_add_library(
    NAME        physics
    SOURCES     ${physics_sources}
    HEADERS     ${physics_headers}
    DEPENDS_ON  mfem integrators numerics infrastructure physics_utilities physics_operators axom fmt mpi
    )

target_include_directories(physics PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${physics_headers} DESTINATION include/physics )

install(TARGETS              physics
        EXPORT               serac-targets
        DESTINATION          lib
        )
