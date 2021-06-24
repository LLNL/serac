# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(numerics_headers
    array_4D.hpp
    expr_template_impl.hpp
    expr_template_ops.hpp
    mesh_utils.hpp
    vector_expression.hpp
    assembled_sparse_matrix.hpp
    )

set(numerics_sources
    mesh_utils.cpp
    assembled_sparse_matrix.cpp	
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
