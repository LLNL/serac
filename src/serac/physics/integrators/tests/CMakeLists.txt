# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies gtest serac_physics_integrators)
blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

set(integrator_tests
    serac_wrapper_tests.cpp)

serac_add_tests( SOURCES ${integrator_tests}
                 DEPENDS_ON ${test_dependencies})    
