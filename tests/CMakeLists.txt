# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

#------------------------------------------------------------------------------
# Tests for Serac objects 
#------------------------------------------------------------------------------

if (ENABLE_GTEST)

    set(test_dependencies solvers coefficients axom mpi mfem gtest)

    set(solver_tests
        serac_dynamic_solver.cpp
        serac_linelastic_solver.cpp
        serac_quasistatic_solver.cpp
        serac_thermal_solver.cpp
        serac_thermal_structural_solver.cpp
        serac_dtor.cpp
        serac_boundary_cond.cpp)

    foreach(filename ${solver_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${test_dependencies}
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 2 )
    endforeach()


    set(coefficent_tests
        serac_component_bc.cpp
        serac_stdfunction_coefficient.cpp
        serac_wrapper_tests.cpp)

    foreach(filename ${coefficent_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${test_dependencies}
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 1 )
    endforeach()

    set(language_tests
        copy_elision.cpp
        expr_templates.cpp
        mfem_array_std_algo.cpp)

    foreach(filename ${language_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${test_dependencies}
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 1 )
    endforeach()

    if(ENABLE_CUDA)
        # CUDA smoke test
        blt_add_library( NAME       serac_cuda_smoketest_kernel
                         SOURCES    serac_cuda_smoketest_kernel.cpp 
                         DEPENDS_ON cuda)
        blt_add_executable( NAME       serac_cuda_smoketest
                            SOURCES    serac_cuda_smoketest.cpp 
                            OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                            DEPENDS_ON serac_cuda_smoketest_kernel ${test_dependencies} cuda_runtime
                            FOLDER      serac/tests)
        blt_add_test( NAME          serac_cuda_smoketest
                      COMMAND       serac_cuda_smoketest )
    endif()

endif()

# Make sure running serac driver doesn't completely fail
blt_add_test(NAME          serac
             COMMAND       serac
             NUM_MPI_TASKS 1 )
