# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(solvers_sources
    base_solver.cpp
    nonlinear_solid_solver.cpp
    nonlinear_solid_operators.cpp
    thermal_solver.cpp
    thermal_operators.cpp
    elasticity_solver.cpp
    )

set(solvers_headers
    base_solver.hpp
    nonlinear_solid_solver.hpp
    nonlinear_solid_operators.hpp
    thermal_solver.hpp
    thermal_operators.hpp
    elasticity_solver.hpp
    )

blt_add_library(
    NAME        solvers
    SOURCES     ${solvers_sources}
    HEADERS     ${solvers_headers}
    DEPENDS_ON  mfem integrators common fmt mpi
    )

target_include_directories(solvers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
