# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# not sure how we want to handle performance-related tests
set(functional_performance_tests
    comparisons_H1.cpp
    comparisons_Hcurl.cpp)
 
foreach(filename ${functional_performance_tests})
  get_filename_component(test_name ${filename} NAME_WE)
  blt_add_executable(NAME        ${test_name}
                     SOURCES     ${filename}
                     OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                     DEPENDS_ON  gtest serac_functional ${functional_depends}
                     FOLDER      serac/tests )
  set_source_files_properties(${filename} PROPERTIES LANGUAGE CXX)
  #target_compile_definitions(${test_name} PUBLIC USE_OLD_IMPLEMENTATION)
endforeach()

target_precompile_headers(comparisons_H1 PUBLIC pch.hpp)

if(ENABLE_CUDA)

    set(functional_performance_tests_cuda 
        sum_factorization_tests.cu # todo: delete this!
        memory_bandwidth_test.cu
        batched_shape_functions_cuda.cu
        mass_diffusion_tests_cuda.cu)

    foreach(filename ${functional_performance_tests_cuda})
        get_filename_component(test_name ${filename} NAME_WE)
        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  gtest serac_functional cuda ${functional_depends}
                           FOLDER      serac/tests )
    endforeach()

    target_compile_options(batched_shape_functions_cuda PRIVATE --extended-lambda)
    target_compile_options(mass_diffusion_tests_cuda PRIVATE --extended-lambda)

  add_executable(sandbox cuda_sandbox.cu)

endif()
