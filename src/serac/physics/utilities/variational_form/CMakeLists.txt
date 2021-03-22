# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(weakform_depends serac_physics mfem mpi)

# Add the library first
set(weakform_headers
    weak_form.hpp
    tensor.hpp
    array.hpp
    dual.hpp
    tuple_arithmetic.hpp
    polynomials.hpp
    quadrature.hpp
    )

blt_add_library(
      NAME        serac_weakform
      HEADERS     ${weakform_headers}
      SOURCES     ${weakform_sources}
      DEPENDS_ON  ${weakform_depends}
      )


install(FILES ${weakform_headers} DESTINATION include/serac/physics/utilities/weak_form )

install(TARGETS              serac_weakform
        EXPORT               serac-targets
        DESTINATION          lib
        )


# Then add the examples/tests
set(weakform_tests_serial
    tests/hcurl_unit_tests.cpp
    tests/test_tensor_ad.cpp
    tests/tuple_arithmetic_unit_tests.cpp
    tests/tensor_unit_tests.cpp)
    
foreach(filename ${weakform_tests_serial})
    get_filename_component(test_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_weakform ${weakform_depends}
                       FOLDER      serac/tests )
    blt_add_test(NAME          ${test_name}
                 COMMAND       ${test_name}
                 NUM_MPI_TASKS 1 )
endforeach()

# Then add the examples/tests
set(weakform_tests_mpi
    tests/integration_rule_problem.cpp
    tests/weak_form_comparisons.cpp)
    
foreach(filename ${weakform_tests_mpi})
    get_filename_component(test_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_weakform ${weakform_depends}
                       FOLDER      serac/tests )
    blt_add_test(NAME          ${test_name}
                 COMMAND       ${test_name}
                 NUM_MPI_TASKS 4)
endforeach()
    