# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(utilities)
add_subdirectory(operators)

set(physics_sources
    base_physics.cpp
    solid.cpp
    thermal_conduction.cpp
    thermal_solid.cpp
    )

set(physics_headers
    base_physics.hpp
    solid.hpp
    thermal_conduction.hpp
    thermal_solid.hpp
    )

blt_add_library(
    NAME        physics
    SOURCES     ${physics_sources}
    HEADERS     ${physics_headers}
    DEPENDS_ON  mfem integrators numerics infrastructure physics_utilities physics_operators axom fmt mpi
    )

target_include_directories(physics PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${physics_headers} DESTINATION include/physics )

install(TARGETS              physics
        EXPORT               serac-targets
        DESTINATION          lib
        )
