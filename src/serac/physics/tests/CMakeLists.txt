# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies serac_physics serac_mesh gtest)

set(serial_solver_tests
    beam_bending.cpp
    contact_patch.cpp
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
    lce_Brighenti_tensile.cpp
    lce_Bertoldi_lattice.cpp
    parameterized_thermomechanics_example.cpp
    parameterized_thermal.cpp
    solid.cpp
    solid_periodic.cpp
    solid_shape.cpp
    thermal_shape.cpp
    thermal_mechanics.cpp
    )

serac_add_tests( SOURCES ${solver_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)
