# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies gtest serac_physics_operators serac_boundary_conditions serac_state)
blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

set(operators_tests
    serac_operator.cpp
    serac_odes.cpp)

serac_add_tests( SOURCES ${operators_tests}
                 DEPENDS_ON ${test_dependencies})
