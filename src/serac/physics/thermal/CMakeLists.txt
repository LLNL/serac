# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(thermal_sources
    thermal_conduction.cpp
    )

set(thermal_headers
    thermal_conduction.hpp
    )

set(thermal_dependencies
    serac_infrastructure
    serac_numerics
    serac_physics_utilities
    serac_physics_operators
    )

blt_add_library(
    NAME        serac_physics_thermal
    SOURCES     ${thermal_sources}
    HEADERS     ${thermal_headers}
    DEPENDS_ON  ${thermal_dependencies}
    )

install(FILES ${thermal_headers} DESTINATION include/serac/physics/thermal )

install(TARGETS              serac_physics_thermal
        EXPORT               serac-targets
        DESTINATION          lib
        )
