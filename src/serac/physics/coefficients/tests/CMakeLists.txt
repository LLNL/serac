# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies serac_physics test_utils)
blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

set(coefficient_tests
    serac_component_bc.cpp)

serac_add_tests( SOURCES ${coefficient_tests}
                 DEPENDS_ON ${test_dependencies})
