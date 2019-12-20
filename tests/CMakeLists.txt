# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

#------------------------------------------------------------------------------
# Tests for Serac solvers 
#------------------------------------------------------------------------------

if (ENABLE_GTEST)

    #----------------------
    # Dynamic Solver
    #----------------------
    blt_add_executable(
        NAME       serac_dynamic_solver
        SOURCES    serac_dynamic_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest )
    blt_add_test(
        NAME              serac_dynamic_solver
        COMMAND           serac_dynamic_solver -m ${PROJECT_SOURCE_DIR}/data/beam-hex.mesh
        NUM_MPI_TASKS     4
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Linear Elasticity Solver
    #----------------------
    blt_add_executable(
        NAME       serac_linelastic_solver
        SOURCES    serac_linelastic_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest )
    blt_add_test(
        NAME              serac_linelastic_solver
        COMMAND           serac_linelastic_solver -m ${PROJECT_SOURCE_DIR}/data/beam-quad.mesh
        NUM_MPI_TASKS     3
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )


    #----------------------
    # Quasistatic Solver
    #----------------------
    blt_add_executable(
        NAME       serac_quasistatic_solver
        SOURCES    serac_quasistatic_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers coefficients mpi mfem gtest )
    blt_add_test(
        NAME              serac_quasistatic_solver
        COMMAND           serac_quasistatic_solver -m ${PROJECT_SOURCE_DIR}/data/beam-hex.mesh
        NUM_MPI_TASKS     4
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Conduction Solver
    #----------------------
    blt_add_executable(
        NAME       serac_conduction_solver
        SOURCES    serac_conduction_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest )
    blt_add_test(
        NAME              serac_conduction_solver
        COMMAND           serac_conduction_solver -m ${PROJECT_SOURCE_DIR}/data/star.mesh
        NUM_MPI_TASKS     4
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

endif()
