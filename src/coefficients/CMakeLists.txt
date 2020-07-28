# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(coefficients_sources
    loading_functions.cpp
    stdfunction_coefficient.cpp
    )

set(coefficients_headers
    loading_functions.hpp
    traction_coefficient.hpp
    stdfunction_coefficient.hpp
    )

blt_add_library(
    NAME        coefficients
    SOURCES     ${coefficients_sources}
    HEADERS     ${coefficients_headers}
    DEPENDS_ON  axom mfem mpi
    )

target_include_directories(coefficients PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
