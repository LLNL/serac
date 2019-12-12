# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

#------------------------------------------------------------------------------
# Tests for Serac solvers 
#------------------------------------------------------------------------------

set( serac_tests
     serac_quasistatic_solver.cpp
     serac_dynamic_solver.cpp
     serac_conduction_solver.cpp )

if (ENABLE_GTEST)
    foreach ( test_source ${serac_tests} )

        get_filename_component( test_name ${test_source} NAME_WE )
        set( test_cmd ${test_name}_test )

        blt_add_executable(
            NAME ${test_cmd}
            SOURCES ${test_source}
            OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
            DEPENDS_ON solvers mpi mfem gtest
        )

        blt_add_test(
            NAME ${test_name}
            COMMAND ${test_cmd}
            NUM_MPI_TASKS 4
            WORKING_DIRECTORY ${TEST_OUTPUT_DIRECTORY}
        )

    endforeach()
endif()

serac_add_code_checks(PREFIX serac_tests)
