# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies serac_physics serac_state test_utils)
blt_list_append( TO test_dependencies ELEMENTS caliper adiak IF ${SERAC_ENABLE_PROFILING} )

blt_add_library(
    NAME        test_utils
    SOURCES     test_utilities.cpp
    HEADERS     test_utilities.hpp
    DEPENDS_ON  serac_physics serac_state gtest
    )

set(serial_solver_tests
    solid_sensitivity.cpp
    thermal_functional_finite_diff.cpp
    solid_functional_finite_diff.cpp
    test_solid_functional_statics.cpp
    )

serac_add_tests( SOURCES ${serial_solver_tests}
    DEPENDS_ON ${test_dependencies}
    NUM_MPI_TASKS 1)

set(solver_tests
    #liquid_crystal_elastomer_example.cpp
    parameterized_thermomechanics_example.cpp
    solid.cpp
    solid_reuse_mesh.cpp
    solid_adjoint.cpp
    solid_functional.cpp
    solid_functional_shape.cpp
    thermal_solver.cpp
    thermal_solid_solver.cpp
    thermal_functional.cpp
    thermal_mechanics_functional.cpp
    )

serac_add_tests( SOURCES ${solver_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)

set(solver_utility_tests
    physics_dtor.cpp
    newmark_test.cpp)

serac_add_tests( SOURCES ${solver_utility_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)
