# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(functional)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(numerics_headers
    equation_solver.hpp
    expr_template_impl.hpp
    expr_template_ops.hpp
    odes.hpp
    quadrature_data.hpp
    solver_config.hpp
    stdfunction_operator.hpp
    vector_expression.hpp
    )

set(numerics_sources
    equation_solver.cpp
    odes.cpp
    )

set(numerics_depends serac_infrastructure)

blt_add_library(
    NAME        serac_numerics
    HEADERS     ${numerics_headers}
    SOURCES     ${numerics_sources}
    DEPENDS_ON  ${numerics_depends}
    )

install(FILES ${numerics_headers} DESTINATION include/serac/numerics )

install(TARGETS              serac_numerics
        EXPORT               serac-targets
        DESTINATION          lib
        )
