# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(functional)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(numerics_headers
    equation_solver.hpp
    odes.hpp
    solver_config.hpp
    stdfunction_operator.hpp
    petsc_solvers.hpp
    )

set(numerics_sources
    equation_solver.cpp
    odes.cpp
    petsc_solvers.cpp
    )

set(numerics_depends serac_infrastructure serac_functional)
blt_list_append(TO numerics_depends ELEMENTS PkgConfig::PETSC IF PETSC_FOUND)

blt_add_library(
    NAME        serac_numerics
    HEADERS     ${numerics_headers}
    SOURCES     ${numerics_sources}
    DEPENDS_ON  ${numerics_depends}
    )

install(FILES ${numerics_headers} DESTINATION include/serac/numerics )

install(TARGETS              serac_numerics
        EXPORT               serac-targets
        DESTINATION          lib
        )
