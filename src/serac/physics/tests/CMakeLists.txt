# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies serac_physics test_utils)
blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

blt_add_library(
    NAME        test_utils
    SOURCES     test_utilities.cpp
    HEADERS     test_utilities.hpp
    DEPENDS_ON  serac_physics gtest
    )

set(solver_tests
    serac_solid.cpp
    serac_solid_reuse_mesh.cpp
    serac_solid_adjoint.cpp
    serac_thermal_solver.cpp
    serac_thermal_structural_solver.cpp
    serac_thermal_expansion.cpp
    )

serac_add_tests( SOURCES ${solver_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)

set(solver_utility_tests
    serac_dtor.cpp
    serac_boundary_cond.cpp
    serac_newmark_test.cpp)

serac_add_tests( SOURCES ${solver_utility_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)
