# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(integrators_sources
    displacement_hyperelastic_integrator.cpp
    nonlinear_reaction_integrator.cpp
    traction_integrator.cpp
    wrapper_integrator.cpp
    )

set(integrators_headers
    displacement_hyperelastic_integrator.hpp
    nonlinear_reaction_integrator.hpp
    traction_integrator.hpp
    wrapper_integrator.hpp
    )

set(integrators_depends
    serac_infrastructure
    serac_numerics
    serac_physics_materials
    )

blt_add_library(
    NAME        serac_physics_integrators
    SOURCES     ${integrators_sources}
    HEADERS     ${integrators_headers}
    DEPENDS_ON  ${integrators_depends}
    )

install(FILES ${integrators_headers} DESTINATION include/serac/physics/integrators )

install(TARGETS              serac_physics_integrators 
        EXPORT               serac-targets
        DESTINATION          lib
        )
