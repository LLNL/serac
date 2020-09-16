# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(math_headers
    expr_template_internal.hpp
    expr_template_ops.hpp
    mesh_utils.hpp
    vector_expression.hpp
    )

set(math_sources
    mesh_utils.cpp
    )

set(math_depends mfem mpi fmt)

blt_add_library(
    NAME        math
    HEADERS     ${math_headers}
    SOURCES     ${math_sources}
    DEPENDS_ON  ${math_depends}
    )


target_include_directories(math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${math_headers} DESTINATION include/math )

install(TARGETS              math
        EXPORT               serac-targets
        DESTINATION          lib
        )
