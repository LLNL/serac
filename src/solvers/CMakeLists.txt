# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(solvers_sources
    base_solver.cpp
    quasistatic_solver.cpp
    dynamic_solver.cpp
    thermal_solver.cpp
    linear_elasticity_solver.cpp
    )

set(solvers_headers
    base_solver.hpp
    quasistatic_solver.hpp
    dynamic_solver.hpp
    thermal_solver.hpp
    linear_elasticity_solver.hpp
    )

blt_add_library(
    NAME        solvers
    SOURCES     ${solvers_sources}
    HEADERS     ${solvers_headers}
    DEPENDS_ON  mfem integrators common mpi
    )

target_include_directories(solvers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
