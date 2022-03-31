# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# This test is all constexpr-evaluated, so it doesn't
# actually need to run, if it compiles without error, the tests have passed
blt_add_executable(NAME        tensor_unit_tests
                   SOURCES     tensor_unit_tests.cpp
                   OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON  gtest serac_functional ${functional_depends}
                   FOLDER      serac/tests)

# Then add the examples/tests
set(functional_tests_serial
    get_qf_derivative_type.cpp
    hcurl_unit_tests.cpp
    test_tensor_ad.cpp
    tuple_arithmetic_unit_tests.cpp)

serac_add_tests( SOURCES ${functional_tests_serial}
                 DEPENDS_ON gtest serac_functional ${functional_depends})

# Then add the examples/tests
set(functional_tests_mpi
    functional_basic.cpp
    functional_multiphysics.cpp
    functional_qoi.cpp
    functional_nonlinear.cpp
    functional_boundary_test.cpp
    functional_comparisons.cpp
    functional_comparison_L2.cpp
    functional_material_state_test.cpp
    )

serac_add_tests( SOURCES ${functional_tests_mpi}
                 DEPENDS_ON gtest serac_functional serac_state ${functional_depends}
                 NUM_MPI_TASKS 4 )

# not sure how we want to handle performance-related tests
set(functional_performance_tests
    performance/J2_material_with_AD.cpp)

# Only add diffusion test for all compilers except for gcc-11 to prevent compiler-related bug
# src/serac/numerics/functional/tests/performance/diffusion_performance_test.cpp:97:24: sorry, unimplemented: unexpected AST of kind nontype_argument_pack
#   97 |     grad[i][j][k] = dot(x, element_type::shape_function_gradients({xi[i], xi[j], xi[k]}));
if(NOT (C_COMPILER_FAMILY_IS_GNU AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER 11.0.0))
    list(APPEND function_performance_tests
         performance/diffusion_performance_test.cpp)
endif()
 
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
        tensor_unit_tests_cuda.cu 
#        some of the GPU functionality is temporarily disabled to 
#        help incrementally roll-out the variadic implementation of Functional
#        TODO: re-enable GPU kernels in a follow-up PR
#        functional_comparisons_cuda.cu
#        quadrature_data_cuda.cu
       )

    serac_add_tests( SOURCES ${functional_tests_cuda}
                     DEPENDS_ON gtest serac_functional serac_state ${functional_depends} cuda)

    # FIXME: Different linker is used depending on whether axom is a dependency??
    # clang directly (with axom) or the MPI wrapper (without axom)
    #serac_add_tests( SOURCES mfem_issue_reproducer.cu
    #                 DEPENDS_ON mfem cuda mpi axom)

    blt_add_executable(NAME        mfem_issue_reproducer
                       SOURCES     mfem_issue_reproducer.cu
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_functional ${functional_depends} cuda
                       FOLDER      serac/tests )


endif()
