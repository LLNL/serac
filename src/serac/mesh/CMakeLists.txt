# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(mesh_headers
    mesh_utils.hpp
    mesh_utils_base.hpp
    )

set(mesh_sources
    mesh_utils.cpp
    )

set(mesh_depends serac_infrastructure)

blt_add_library(
    NAME        serac_mesh
    HEADERS     ${mesh_headers}
    SOURCES     ${mesh_sources}
    DEPENDS_ON  ${mesh_depends}
    )

install(FILES ${mesh_headers} DESTINATION include/serac/mesh )

install(TARGETS              serac_mesh
        EXPORT               serac-targets
        DESTINATION          lib
        )
