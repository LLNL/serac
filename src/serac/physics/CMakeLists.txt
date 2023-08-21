# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(materials)
add_subdirectory(state)
add_subdirectory(boundary_conditions)

if(SERAC_USE_TRIBOL)
    add_subdirectory(contact)
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(SERAC_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

set(physics_sources
    base_physics.cpp
    heat_transfer_input.cpp
    solid_mechanics_input.cpp
    thermomechanics_input.cpp
    )

set(physics_headers
    base_physics.hpp
    common.hpp
    heat_transfer.hpp
    heat_transfer_input.hpp
    solid_mechanics.hpp
    solid_mechanics_input.hpp
    thermomechanics.hpp
    thermomechanics_input.hpp
    )

set(physics_dependencies
    serac_state
    serac_boundary_conditions
    serac_numerics
    )
blt_list_append(TO physics_dependencies ELEMENTS serac_contact IF SERAC_USE_TRIBOL)

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
