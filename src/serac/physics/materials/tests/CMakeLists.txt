# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(test_dependencies serac_physics_materials gtest)

set(material_tests
    thermomechanical_material.cpp
    J2_material.cpp
    nonlinear_J2_material.cpp
    parameterized_nonlinear_J2_material.cpp
    liquidCrystalElastomer_material.cpp
)

serac_add_tests( SOURCES ${material_tests}
                 DEPENDS_ON ${test_dependencies})
 
