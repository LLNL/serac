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
    functional_shape_derivatives.cpp
    simplex_basis_function_unit_tests.cpp
    domain_tests.cpp
    geometric_factors_tests.cpp
    hcurl_unit_tests.cpp
    test_tensor_ad.cpp
    tuple_arithmetic_unit_tests.cpp
    test_newton.cpp)

serac_add_tests( SOURCES ${functional_tests_serial}
                 DEPENDS_ON gtest serac_functional ${functional_depends})

# Then add the examples/tests
set(functional_tests_mpi
    #functional_basic_hcurl.cpp
    functional_with_domain.cpp
    functional_basic_h1_scalar.cpp
    functional_basic_h1_vector.cpp
    functional_multiphysics.cpp
    functional_qoi.cpp
    functional_nonlinear.cpp
    functional_boundary_test.cpp
    functional_comparisons.cpp
    functional_comparison_L2.cpp
    )

serac_add_tests( SOURCES ${functional_tests_mpi}
                 DEPENDS_ON gtest serac_functional serac_state ${functional_depends}
                 NUM_MPI_TASKS 4 )

foreach(filename ${functional_tests_mpi})
    set_source_files_properties(${filename} PROPERTIES LANGUAGE CXX)
endforeach()

if(ENABLE_CUDA)

    set(functional_tests_cuda 
        tensor_unit_tests_cuda.cu 
#        some of the GPU functionality is temporarily disabled to 
#        help incrementally roll-out the variadic implementation of Functional
#        TODO: re-enable GPU kernels in a follow-up PR
#        functional_comparisons_cuda.cu
       )

    serac_add_tests( SOURCES ${functional_tests_cuda}
                     DEPENDS_ON gtest serac_functional serac_state ${functional_depends})

endif()
