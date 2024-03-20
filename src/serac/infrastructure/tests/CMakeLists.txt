# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies gtest serac_physics serac_mesh)

set(infrastructure_tests
    error_handling.cpp
    input.cpp
    profiling.cpp)

serac_add_tests( SOURCES ${infrastructure_tests}
                 DEPENDS_ON ${test_dependencies})
