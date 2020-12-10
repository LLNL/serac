# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(utilities)
add_subdirectory(operators)

set(physics_sources
    base_physics.cpp
    nonlinear_solid.cpp
    thermal_conduction.cpp
    thermal_solid.cpp
    elasticity.cpp
    )

set(physics_headers
    base_physics.hpp
    nonlinear_solid.hpp
    thermal_conduction.hpp
    thermal_solid.hpp
    elasticity.hpp
    )

blt_add_library(
    NAME        physics
    SOURCES     ${physics_sources}
    HEADERS     ${physics_headers}
    DEPENDS_ON  mfem integrators numerics infrastructure physics_utilities physics_operators axom fmt mpi
    )

install(FILES ${physics_headers} DESTINATION include/serac/physics )

install(TARGETS              physics
        EXPORT               serac-targets
        DESTINATION          lib
        )
