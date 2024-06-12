# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies serac_physics serac_mesh gtest)

set(serial_solver_tests
    beam_bending.cpp
    thermal_finite_diff.cpp
    thermal_statics_patch.cpp
    thermal_dynamics_patch.cpp
    solid_finite_diff.cpp
    solid_statics_patch.cpp
    solid_dynamics_patch.cpp
    solid_rigid_contact_adjoint.cpp
    dynamic_solid_adjoint.cpp
    finite_element_vector_set_over_domain.cpp
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
    solid_robin_condition.cpp
    thermal_shape.cpp
    thermal_mechanics.cpp
    thermal_robin_condition.cpp
    dynamic_thermal_adjoint.cpp
    solid_reaction_adjoint.cpp
    thermal_nonlinear_solve.cpp
    solid_nonlinear_solve.cpp
    )
blt_list_append(TO solver_tests ELEMENTS contact_patch.cpp contact_beam.cpp IF TRIBOL_FOUND)

serac_add_tests( SOURCES ${solver_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)
