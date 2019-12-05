# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(solvers_sources
    quasistatic_solver.cpp
    dynamic_solver.cpp
    conduction_solver.cpp
    )

set(solvers_headers
    quasistatic_solver.hpp
    dynamic_solver.hpp
    conduction_solver.hpp
    )

blt_add_library(
    NAME        solvers
    SOURCES     ${solvers_sources}
    HEADERS     ${solvers_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis integrators linalg
    )

target_include_directories(solvers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
