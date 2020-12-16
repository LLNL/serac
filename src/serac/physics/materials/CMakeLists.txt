# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(materials_headers
    hyperelastic_material.hpp
    )

set(materials_sources
    hyperelastic_material.cpp
    )

set(materials_depends
    serac_infrastructure
    )

blt_add_library(
   NAME        serac_materials
   HEADERS     ${materials_headers}
   SOURCES     ${materials_sources}
   DEPENDS_ON  ${materials_depends}
   )

install(FILES ${materials_headers} DESTINATION include/serac/physics/materials )

install(TARGETS              serac_materials
        EXPORT               serac-targets
        DESTINATION          lib
        )