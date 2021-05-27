# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(weakform_depends serac_physics mfem mpi)

#if(ENABLE_CUDA)
#    list(APPEND weakform_depends cuda)
#endif()

# Add the library first
set(weakform_headers
    weak_form.hpp
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
      NAME        serac_weakform
      HEADERS     ${weakform_headers}
      DEPENDS_ON  ${weakform_depends}
      )


install(FILES ${weakform_headers} DESTINATION include/serac/physics/utilities/weak_form )

install(TARGETS              serac_weakform
        EXPORT               serac-targets
        DESTINATION          lib
        )

# this test is all constexpr-evaluated, so it doesn't
# actually need to run, if it compiles without error, the tests have passed
blt_add_executable(NAME        tensor_unit_tests
                   SOURCES     tests/tensor_unit_tests.cpp
                   OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON  gtest serac_weakform ${weakform_depends}
                   FOLDER      serac/tests)

if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
