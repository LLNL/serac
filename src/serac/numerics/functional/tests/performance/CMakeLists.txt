# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# not sure how we want to handle performance-related tests
set(functional_performance_tests
    batched_shape_functions_scalar.cpp
    J2_material_with_AD.cpp
    diffusion_performance_test.cpp)
 
foreach(filename ${functional_performance_tests})
    get_filename_component(test_name ${filename} NAME_WE)
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_functional ${functional_depends}
                       FOLDER      serac/tests )
endforeach()

if(ENABLE_CUDA)

    set(functional_performance_tests_cuda 
        batched_shape_functions_scalar_cuda.cu)

    foreach(filename ${functional_performance_tests_cuda})
        get_filename_component(test_name ${filename} NAME_WE)
        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  gtest serac_functional cuda ${functional_depends}
                           FOLDER      serac/tests )
    endforeach()

    target_compile_options(batched_shape_functions_scalar_cuda PRIVATE --extended-lambda)

endif()
