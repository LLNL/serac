# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(physics_operators_headers
    nonlinear_solid_operators.hpp
    thermal_operators.hpp
    )

set(physics_operators_sources
    nonlinear_solid_operators.cpp
    thermal_operators.cpp
    )

set(physics_operators_depends infrastructure physics_utilities numerics fmt mfem mpi axom)

blt_add_library(
    NAME        physics_operators
    HEADERS     ${physics_operators_headers}
    SOURCES     ${physics_operators_sources}
    DEPENDS_ON  ${physics_operators_depends}
    )


target_include_directories(physics_operators PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
        )

install(FILES ${physics_operators_headers} DESTINATION include/physics/operators )

install(TARGETS              physics_operators
        EXPORT               serac-targets
        DESTINATION          lib
        )
