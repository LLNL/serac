# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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
    thermal_statics_patch.cpp
    thermal_dynamics_patch.cpp
    solid_finite_diff.cpp
    solid_statics_patch.cpp
    solid_dynamics_patch.cpp
    )

serac_add_tests( SOURCES ${serial_solver_tests}
    DEPENDS_ON ${test_dependencies}
    NUM_MPI_TASKS 1)

set(solver_tests
    lce_Brighenti_logpile.cpp
    lce_Brighenti_free_swelling.cpp
    lce_Brighenti_tensile.cpp
    lce_Brighenti_compression.cpp
    lce_Brighenti.cpp
    lce_Bertoldi_free_swelling.cpp
    lce_Bertoldi_tensile.cpp
    lce_Bertoldi_tensile_new_data.cpp
    lce_Bertoldi_rectLattice.cpp
    lce_Bertoldi_honeycomb_lattice.cpp
    lce_Bertoldi.cpp
    parameterized_thermomechanics_example.cpp
    parameterized_thermal.cpp
    solid_legacy.cpp
    solid_legacy_reuse_mesh.cpp
    solid_legacy_adjoint.cpp
    solid.cpp
    solid_periodic.cpp
    solid_shape.cpp
    thermal_legacy.cpp
    thermal_shape.cpp
    thermal_solid_legacy.cpp
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
