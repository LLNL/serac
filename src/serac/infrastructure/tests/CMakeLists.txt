# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(infrastructure_test_depends serac_physics serac_mesh gtest)

set(infrastructure_test_sources
    error_handling.cpp
    input.cpp
    profiling.cpp)

serac_add_tests( SOURCES    ${infrastructure_test_sources}
                 DEPENDS_ON ${infrastructure_test_depends})
