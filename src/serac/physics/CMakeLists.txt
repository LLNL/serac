# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(integrators)
add_subdirectory(materials)
add_subdirectory(state)
add_subdirectory(boundary_conditions)
add_subdirectory(coefficients)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(SERAC_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

set(physics_sources
    base_physics.cpp
    solid_legacy.cpp
    thermal_conduction_legacy.cpp
    thermal_solid_legacy.cpp
    )

set(physics_headers
    base_physics.hpp
    common.hpp
    solid_legacy.hpp
    solid_mechanics.hpp
    thermal_conduction_legacy.hpp
    heat_transfer.hpp
    thermomechanics.hpp
    thermal_solid_legacy.hpp
    )

set(physics_dependencies
    serac_coefficients
    serac_physics_integrators
    serac_state
    serac_boundary_conditions
    serac_numerics
    )

blt_add_library(
    NAME        serac_physics
    SOURCES     ${physics_sources}
    HEADERS     ${physics_headers}
    DEPENDS_ON  ${physics_dependencies}
    )

install(FILES ${physics_headers} DESTINATION include/serac/physics )

install(TARGETS              serac_physics
        EXPORT               serac-targets
        DESTINATION          lib
        )
