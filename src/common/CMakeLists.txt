# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(common_headers
    expr_template_internal.hpp
    expr_template_ops.hpp
    serac_types.hpp
    mesh_utils.hpp
    vector_expression.hpp
    )

set(common_sources
    mesh_utils.cpp
    )

set(common_depends mfem mpi fmt)

blt_add_library(
    NAME        common
    HEADERS     ${common_headers}
    SOURCES     ${common_sources}
    DEPENDS_ON  ${common_depends}
    )


target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
