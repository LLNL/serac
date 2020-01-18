# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(common_headers
    serac_types.hpp
    )

blt_add_library(
    NAME        common
    HEADERS     ${common_headers}
    )

target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
