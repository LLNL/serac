# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(boundary_conditions_headers
    boundary_condition.hpp
    boundary_condition_helper.hpp
    boundary_condition_manager.hpp
    )

set(boundary_conditions_sources
    boundary_condition.cpp
    boundary_condition_helper.cpp
    boundary_condition_manager.cpp
    )

set(boundary_conditions_depends serac_state)

blt_add_library(
    NAME        serac_boundary_conditions
    HEADERS     ${boundary_conditions_headers}
    SOURCES     ${boundary_conditions_sources}
    DEPENDS_ON  ${boundary_conditions_depends}
    )

install(FILES ${boundary_conditions_headers} DESTINATION include/serac/physics/boundary_conditions )

install(TARGETS              serac_boundary_conditions
        EXPORT               serac-targets
        DESTINATION          lib
        )
