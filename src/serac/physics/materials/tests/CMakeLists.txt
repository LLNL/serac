# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(test_dependencies gtest serac_physics test_utils)

set(material_tests
    thermomechanical_material.cpp
    )

serac_add_tests( SOURCES ${material_tests}
                 DEPENDS_ON ${test_dependencies})
 
