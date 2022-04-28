# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(functional_depends mfem mpi serac_infrastructure serac_mesh)
blt_list_append( TO functional_depends ELEMENTS cuda    IF ENABLE_CUDA )
blt_list_append( TO functional_depends ELEMENTS caliper adiak IF SERAC_ENABLE_PROFILING )

# Add the library first
set(functional_headers
    boundary_integral.hpp
    boundary_integral_kernels.hpp
    dof_numbering.hpp
    domain_integral.hpp
    domain_integral_kernels.hpp
    dual.hpp
    evector_view.hpp
    finite_element.hpp
    functional.hpp
    functional_qoi.inl
    integral_utilities.hpp
    isotropic_tensor.hpp
    polynomials.hpp
    quadrature.hpp
    tensor.hpp
    tuple.hpp
    tuple_arithmetic.hpp
    )

set(functional_detail_headers
    detail/hexahedron_H1.inl
    detail/hexahedron_Hcurl.inl
    detail/hexahedron_L2.inl
    detail/metaprogramming.hpp
    detail/qoi.inl
    detail/quadrilateral_H1.inl
    detail/quadrilateral_Hcurl.inl
    detail/quadrilateral_L2.inl
    detail/segment_H1.inl
    detail/segment_Hcurl.inl
    detail/segment_L2.inl
    )

set(functional_cuda_headers
    boundary_integral_kernels.cuh
    domain_integral_kernels.cuh
    )

blt_list_append(TO functional_headers ELEMENTS ${functional_cuda_headers} IF ENABLE_CUDA)

blt_add_library(
      NAME        serac_functional
      HEADERS     ${functional_headers} ${functional_detail_headers}
      DEPENDS_ON  ${functional_depends}
      )


install(FILES ${functional_headers} DESTINATION include/serac/numerics/functional )
install(FILES ${functional_detail_headers} DESTINATION include/serac/numerics/functional/detail )

install(TARGETS              serac_functional
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
