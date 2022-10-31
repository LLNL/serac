# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies test_utils)
blt_list_append( TO test_dependencies ELEMENTS caliper adiak IF ${SERAC_ENABLE_PROFILING} )

blt_add_library(
    NAME        test_utils
    SOURCES     test_utilities.cpp
    HEADERS     test_utilities.hpp
    DEPENDS_ON  serac_physics gtest
    )

set(serial_solver_tests
    solid_legacy_sensitivity.cpp
    thermal_finite_diff.cpp
    solid_finite_diff.cpp
    solid_statics_patch.cpp
    solid_dynamics_patch.cpp
    )

serac_add_tests( SOURCES ${serial_solver_tests}
    DEPENDS_ON ${test_dependencies}
    NUM_MPI_TASKS 1)

set(solver_tests
    liquid_crystal_elastomer_logpile_test.cpp
    liquid_crystal_elastomer_free_swelling_test.cpp
    liquid_crystal_elastomer_tensile_test.cpp
    liquid_crystal_elastomer_test.cpp
    parameterized_thermomechanics_example.cpp
    solid_legacy.cpp
    solid_legacy_reuse_mesh.cpp
    solid_legacy_adjoint.cpp
    solid.cpp
    solid_shape.cpp
    thermal_legacy.cpp
    thermal_solid_legacy.cpp
    thermal.cpp
    thermal_mechanics.cpp
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
