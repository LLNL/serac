# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(material_test_depends serac_physics_materials gtest)

set(material_test_sources
    thermomechanical_material.cpp
    J2_material.cpp
    parameterized_nonlinear_J2_material.cpp
)

serac_add_tests( SOURCES    ${material_test_sources}
                 DEPENDS_ON ${material_test_depends})
