# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(infrastructure_test_dependencies gtest serac_infrastructure)
blt_list_append( TO test_dependencies ELEMENTS caliper adiak IF ${SERAC_ENABLE_PROFILING} )

set(infrastructure_tests
    error_handling.cpp
    input.cpp
    profiling.cpp)

serac_add_tests( SOURCES ${infrastructure_tests}
                 DEPENDS_ON ${infrastructure_test_dependencies})
