# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# not sure how we want to handle performance-related tests
set(benchmark_performance_tests
    performance/J2_material_with_AD.cpp
    performance/diffusion_performance_test.cpp)
 
foreach(filename ${benchmark_performance_tests})
    get_filename_component(test_name ${filename} NAME_WE)
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_functional ${functional_depends}
                       FOLDER      serac/tests )
endforeach()

if(ENABLE_CUDA)

    set(benchmarks_cuda
        benchmark_tensor_unit_tests_cuda.cu)

    serac_add_tests( SOURCES ${benchmarks_cuda}
                     DEPENDS_ON gtest serac_functional ${functional_depends} cuda)



endif()
