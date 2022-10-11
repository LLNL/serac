# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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
    solid.cpp
    thermal_conduction.cpp
    thermal_solid.cpp
    )

set(physics_headers
    base_physics.hpp
    common.hpp
    solid.hpp
    solid_functional.hpp
    thermal_conduction.hpp
    thermal_conduction_functional.hpp
    thermal_mechanics_functional.hpp
    thermal_solid.hpp
    )

set(physics_dependencies
    serac_coefficients
    serac_functional
    serac_physics_integrators
    serac_physics_materials
    serac_boundary_conditions
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
