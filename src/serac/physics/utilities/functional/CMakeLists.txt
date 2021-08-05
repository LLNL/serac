# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(functional_depends serac_physics mfem mpi)
blt_list_append( TO functional_depends ELEMENTS cuda    IF ENABLE_CUDA )
blt_list_append( TO functional_depends ELEMENTS caliper IF SERAC_USE_CALIPER )

# Add the library first
set(functional_headers
    functional.hpp
    tensor.hpp
    dual.hpp
    tuple_arithmetic.hpp
    polynomials.hpp
    quadrature.hpp
    finite_element.hpp
    )

set(functional_detail_headers
    detail/hexahedron_H1.inl
    detail/hexahedron_Hcurl.inl
    detail/hexahedron_L2.inl
    detail/quadrilateral_H1.inl
    detail/quadrilateral_Hcurl.inl
    detail/quadrilateral_L2.inl
    detail/segment_H1.inl
    detail/segment_Hcurl.inl
    detail/segment_L2.inl
    detail/metaprogramming.hpp
    )

blt_add_library(
      NAME        serac_functional
      HEADERS     ${functional_headers} ${functional_detail_headers}
      DEPENDS_ON  ${functional_depends}
      )


install(FILES ${functional_headers} DESTINATION include/serac/physics/utilities/functional )
install(FILES ${functional_detail_headers} DESTINATION include/serac/physics/utilities/functional/detail )

install(TARGETS              serac_functional
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
