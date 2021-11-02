# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_library(
    NAME        serac_precompiled_headers
    SOURCES     precompiled_headers.cpp
    DEPENDS_ON  mfem axom
    )
target_precompile_headers(serac_precompiled_headers PRIVATE precompiled_headers.hpp)

add_subdirectory(infrastructure)
add_subdirectory(numerics)
add_subdirectory(physics)
add_subdirectory(mesh)
