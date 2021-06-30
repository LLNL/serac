# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Then add the examples/tests
set(functional_tests_serial
    hcurl_unit_tests.cpp
    test_tensor_ad.cpp
    tuple_arithmetic_unit_tests.cpp)
    
foreach(filename ${functional_tests_serial})
    get_filename_component(test_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_functional ${functional_depends}
                       FOLDER      serac/tests )
    blt_add_test(NAME          ${test_name}
                 COMMAND       ${test_name}
                 NUM_MPI_TASKS 1 )
endforeach()

# Then add the examples/tests
set(functional_tests_mpi
    # functional_boundary_test.cpp
    functional_comparisons.cpp
    functional_comparison_L2.cpp
    )
    
foreach(filename ${functional_tests_mpi})
    get_filename_component(test_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_functional ${functional_depends}
                       FOLDER      serac/tests )
    blt_add_test(NAME          ${test_name}
                 COMMAND       ${test_name}
                 NUM_MPI_TASKS 4)
endforeach()


# not sure how we want to handle performance-related tests
set(functional_performance_tests
    performance/J2_material_with_AD.cpp
    performance/diffusion_performance_test.cpp)
 
foreach(filename ${functional_performance_tests})
    get_filename_component(test_name ${filename} NAME_WE)
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_functional ${functional_depends}
                       FOLDER      serac/tests )
endforeach()

if(ENABLE_CUDA)

    set(functional_tests_cuda
        tensor_unit_tests_cuda.cu)
        
    foreach(filename ${functional_tests_cuda})
        get_filename_component(test_name ${filename} NAME_WE)
        
        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  gtest serac_functional ${functional_depends} cuda
                           FOLDER      serac/tests)
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 1)
    endforeach()

endif()
