# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies serac_physics test_utils)
blt_list_append( TO test_dependencies ELEMENTS caliper adiak IF ${SERAC_ENABLE_PROFILING} )

blt_add_library(
    NAME        test_utils
    SOURCES     test_utilities.cpp
    HEADERS     test_utilities.hpp
    DEPENDS_ON  serac_physics serac_state gtest
    )

set(serial_solver_tests
    serac_solid_sensitivity.cpp
    serac_thermal_functional_finite_diff.cpp
    serac_solid_functional_finite_diff.cpp
    )

serac_add_tests( SOURCES ${serial_solver_tests}
    DEPENDS_ON ${test_dependencies}
    NUM_MPI_TASKS 1)

set(solver_tests
    serac_solid.cpp
    serac_solid_reuse_mesh.cpp
    serac_solid_adjoint.cpp
    serac_solid_functional.cpp
    serac_thermal_solver.cpp
    serac_thermal_structural_solver.cpp
    serac_thermal_solid_solver.cpp
    serac_thermal_expansion.cpp
    serac_thermal_functional.cpp
    serac_thermal_solid_functional.cpp
    )

serac_add_tests( SOURCES ${solver_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)

set(solver_utility_tests
    serac_dtor.cpp
    serac_newmark_test.cpp)

serac_add_tests( SOURCES ${solver_utility_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)
