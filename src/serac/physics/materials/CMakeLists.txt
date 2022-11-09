# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(materials_sources
    hyperelastic_legacy_material.cpp
    thermal_expansion_legacy_material.cpp
    solid_legacy_utils.cpp
    )

set(materials_headers
    # liquid_crystal_elastomer.hpp
    liquid_crystal_elastomer_material.hpp
    # material_utils.hpp
    green_saint_venant_thermoelastic.hpp
    hyperelastic_legacy_material.hpp
    material_verification_tools.hpp
    parameterized_solid_material.hpp
    parameterized_thermal_material.hpp
    solid_material.hpp
    solid_legacy_utils.hpp
    thermal_expansion_legacy_material.hpp
    thermal_material.hpp
    )

set(materials_depends
    serac_functional
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
