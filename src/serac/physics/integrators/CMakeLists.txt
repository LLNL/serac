# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(integrators_sources
    hyperelastic_traction_integrator.cpp
    displacement_hyperelastic_integrator.cpp
    wrapper_integrator.cpp
    )

set(integrators_headers
    hyperelastic_traction_integrator.hpp
    displacement_hyperelastic_integrator.hpp
    wrapper_integrator.hpp
    )

set(integrators_depends serac_infrastructure)

blt_add_library(
    NAME        serac_integrators
    SOURCES     ${integrators_sources}
    HEADERS     ${integrators_headers}
    DEPENDS_ON  ${integrators_depends}
    )

install(FILES ${integrators_headers} DESTINATION include/serac/physics/integrators )

install(TARGETS              serac_integrators 
        EXPORT               serac-targets
        DESTINATION          lib
        )
