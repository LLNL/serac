# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(thermal_solid_sources
    thermal_solid.cpp
    )

set(thermal_solid_headers
    thermal_solid.hpp
    )

set(thermal_solid_dependencies
    serac_infrastructure
    serac_numerics
    serac_physics_utilities
    serac_physics_operators
    )

blt_add_library(
    NAME        serac_physics_thermal_solid
    SOURCES     ${thermal_solid_sources}
    HEADERS     ${thermal_solid_headers}
    DEPENDS_ON  ${thermal_solid_dependencies}
    )

install(FILES ${thermal_solid_headers} DESTINATION include/serac/physics/thermal_solid )

install(TARGETS              serac_physics_thermal_solid
        EXPORT               serac-targets
        DESTINATION          lib
        )
