# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(solid_sources
    nonlinear_solid.cpp
    elasticity.cpp
    inc_hyperelastic_integrator.cpp
    hyperelastic_traction_integrator.cpp
    )

set(solid_headers
    nonlinear_solid.hpp
    elasticity.hpp
    inc_hyperelastic_integrator.hpp
    hyperelastic_traction_integrator.hpp
    )

set(solid_dependencies
    serac_infrastructure
    serac_numerics
    serac_physics_utilities
    serac_physics_operators
    )

blt_add_library(
    NAME        serac_physics_solid
    SOURCES     ${solid_sources}
    HEADERS     ${solid_headers}
    DEPENDS_ON  ${solid_dependencies}
    )

install(FILES ${solid_headers} DESTINATION include/serac/physics/solid )

install(TARGETS              serac_physics_solid
        EXPORT               serac-targets
        DESTINATION          lib
        )
