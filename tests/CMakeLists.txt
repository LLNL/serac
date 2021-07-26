# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

if (ENABLE_GTEST)

    set(test_dependencies serac_physics serac_coefficients test_utils)
    blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

    set(mfem_tests
        mfem_ex9p_blockilu.cpp
        )

    foreach(filename ${mfem_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  gtest serac_physics_utilities
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 2 )
    endforeach()

    set(language_tests
        copy_elision.cpp
        mfem_array_std_algo.cpp)

    foreach(filename ${language_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  gtest mfem mpi 
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
        blt_add_executable( NAME        serac_cuda_smoketest
                            SOURCES     serac_cuda_smoketest.cpp 
                            OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                            DEPENDS_ON  serac_cuda_smoketest_kernel gtest
                            FOLDER      serac/tests)
        blt_add_test( NAME          serac_cuda_smoketest
                      COMMAND       serac_cuda_smoketest )
    endif()

    if(SERAC_USE_PETSC)
        blt_add_executable( NAME        petsc_smoketest
                            SOURCES     petsc_smoketest.cpp
                            OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                            DEPENDS_ON  PETSc gtest
                            FOLDER      serac/tests )
        blt_add_test(NAME         petsc_smoketest
                    COMMAND       petsc_smoketest
                    NUM_MPI_TASKS 4 )
    endif()

endif()
