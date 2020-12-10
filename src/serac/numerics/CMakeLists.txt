# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(numerics_headers
    expr_template_internal.hpp
    expr_template_ops.hpp
    mesh_utils.hpp
    vector_expression.hpp
    )

set(numerics_sources
    mesh_utils.cpp
    )

set(numerics_depends infrastructure axom mfem mpi fmt)

blt_add_library(
    NAME        numerics
    HEADERS     ${numerics_headers}
    SOURCES     ${numerics_sources}
    DEPENDS_ON  ${numerics_depends}
    )


target_include_directories(numerics PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${numerics_headers} DESTINATION include/serac/numerics )

install(TARGETS              numerics
        EXPORT               serac-targets
        DESTINATION          lib
        )
