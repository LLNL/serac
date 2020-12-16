# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(materials_sources
    hyperelastic_material.cpp
    )

set(materials_headers
    hyperelastic_material.hpp
    )

set(materials_depends infrastructure mfem mpi)
blt_list_append( TO materials_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

blt_add_library(
    NAME        materials
    SOURCES     ${materials_sources}
    HEADERS     ${materials_headers}
    DEPENDS_ON  ${materials_depends}
    )

target_include_directories(materials PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${materials_headers} DESTINATION include/materials )

install(TARGETS              materials 
        EXPORT               serac-targets
        DESTINATION          lib
        )
