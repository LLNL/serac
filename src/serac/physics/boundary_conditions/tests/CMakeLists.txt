# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(boundary_cond_test_depends serac_boundary_conditions gtest)

set(boundary_cond_test_sources
    boundary_cond.cpp)

serac_add_tests( SOURCES       ${boundary_cond_test_sources}
                 DEPENDS_ON    ${boundary_cond_test_depends}
                 NUM_MPI_TASKS 2)
