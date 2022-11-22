# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(physics_core_sources
    solid_mechanics_input.cpp
    )

set(physics_core_headers
    common.hpp
    solid_mechanics_input.hpp
    )

set(physics_core_dependencies
    serac_numerics
    )

blt_add_library(
    NAME        serac_physics_core
    SOURCES     ${physics_core_sources}
    HEADERS     ${physics_core_headers}
    DEPENDS_ON  ${physics_core_dependencies}
    )

install(FILES ${physics_core_headers} DESTINATION include/serac/core )

install(TARGETS              serac_physics_core
        EXPORT               serac-targets
        DESTINATION          lib
        )
