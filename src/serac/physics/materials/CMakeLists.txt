# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(materials_sources
    hyperelastic_material.cpp
    thermal_expansion_material.cpp
    solid_utils.cpp
    )

set(materials_headers
    hyperelastic_material.hpp
    thermal_expansion_material.hpp
    solid_utils.hpp
    thermal_functional_material.hpp
    )

set(materials_depends
    serac_infrastructure
    serac_numerics
    mfem
    )

blt_add_library(
   NAME        serac_physics_materials
   HEADERS     ${materials_headers}
   SOURCES     ${materials_sources}
   DEPENDS_ON  ${materials_depends}
   )

install(FILES ${materials_headers} DESTINATION include/serac/physics/materials )

install(TARGETS              serac_physics_materials
        EXPORT               serac-targets
        DESTINATION          lib
        )
