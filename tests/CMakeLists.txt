# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

#------------------------------------------------------------------------------
# Tests for Serac objects 
#------------------------------------------------------------------------------

if (ENABLE_GTEST)

    #----------------------
    # Dynamic Solver
    #----------------------
    blt_add_executable(
        NAME       serac_dynamic_solver
        SOURCES    serac_dynamic_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest hdf5)
    blt_add_test(
        NAME              serac_dynamic_solver
        COMMAND           serac_dynamic_solver
        NUM_MPI_TASKS     2
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Linear Elasticity Solver
    #----------------------
    blt_add_executable(
        NAME       serac_linelastic_solver
        SOURCES    serac_linelastic_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_linelastic_solver
        COMMAND           serac_linelastic_solver
        NUM_MPI_TASKS     2
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )


    #----------------------
    # Quasistatic Solver
    #----------------------
    blt_add_executable(
        NAME       serac_quasistatic_solver
        SOURCES    serac_quasistatic_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers coefficients mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_quasistatic_solver
        COMMAND           serac_quasistatic_solver
        NUM_MPI_TASKS     2
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Thermal Solver
    #----------------------
    blt_add_executable(
        NAME       serac_thermal_solver
        SOURCES    serac_thermal_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_thermal_solver
        COMMAND           serac_thermal_solver
        NUM_MPI_TASKS     2
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Thermal Structural Solver
    #----------------------
    blt_add_executable(
        NAME       serac_thermal_structural_solver
        SOURCES    serac_thermal_structural_solver.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest coefficients hdf5 )
    blt_add_test(
        NAME              serac_thermal_structural_solver
        COMMAND           serac_thermal_structural_solver
        NUM_MPI_TASKS     2
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Deconstructor test
    #----------------------
    blt_add_executable(
        NAME       serac_dtor
        SOURCES    serac_dtor.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_dtor
        COMMAND           serac_dtor
        NUM_MPI_TASKS     2
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # StdFunctionCoefficient.test
    #----------------------
    blt_add_executable(
        NAME       serac_stdfunction_coefficient
        SOURCES    serac_stdfunction_coefficient.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers coefficients mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_stdfunction_coefficient
        COMMAND           serac_stdfunction_coefficient 
        NUM_MPI_TASKS     1
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Wrapper Unit tests
    #----------------------
    blt_add_executable(
        NAME       serac_wrapper_tests
        SOURCES    serac_wrapper_tests.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers coefficients mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_wrapper_tests
        COMMAND           serac_wrapper_tests
        NUM_MPI_TASKS     1
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

    #----------------------
    # Component-wise BC test
    #----------------------
    blt_add_executable(
        NAME       serac_component_bc
        SOURCES    serac_component_bc.cpp
        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON solvers coefficients mpi mfem gtest hdf5 )
    blt_add_test(
        NAME              serac_component_bc
        COMMAND           serac_component_bc
        NUM_MPI_TASKS     1
        WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY} )

endif()
