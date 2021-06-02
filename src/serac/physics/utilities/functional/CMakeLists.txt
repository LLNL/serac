# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(functional_depends serac_physics mfem mpi)
blt_list_append( TO functional_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

#if(ENABLE_CUDA)
#    list(APPEND functional_depends cuda)
#endif()

# Add the library first
set(functional_headers
    functional.hpp
    tensor.hpp
    dual.hpp
    tuple_arithmetic.hpp
    polynomials.hpp
    quadrature.hpp
    finite_element.hpp
    detail/hexahedron_h1.inl
    detail/hexahedron_hcurl.inl
    detail/quadrilateral_h1.inl
    detail/quadrilateral_hcurl.inl
    detail/quadrilateral_L2.inl
    detail/segment_h1.inl
    detail/metaprogramming.hpp
    )

blt_add_library(
      NAME        serac_functional
      HEADERS     ${functional_headers}
      DEPENDS_ON  ${functional_depends}
      )


install(FILES ${functional_headers} DESTINATION include/serac/physics/utilities/functional )

install(TARGETS              serac_functional
        EXPORT               serac-targets
        DESTINATION          lib
        )

# this test is all constexpr-evaluated, so it doesn't
# actually need to run, if it compiles without error, the tests have passed
blt_add_executable(NAME        tensor_unit_tests
                   SOURCES     tests/tensor_unit_tests.cpp
                   OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON  gtest serac_functional ${functional_depends}
                   FOLDER      serac/tests)

if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
