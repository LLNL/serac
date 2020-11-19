# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(coefficients_sources
    loading_functions.cpp
    coefficient_extensions.cpp
    )

set(coefficients_headers
    loading_functions.hpp
    traction_coefficient.hpp
    coefficient_extensions.hpp
    )

blt_add_library(
    NAME        coefficients
    SOURCES     ${coefficients_sources}
    HEADERS     ${coefficients_headers}
    DEPENDS_ON  axom mfem mpi
    )

target_include_directories(coefficients PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${coefficients_headers} DESTINATION include/coefficients )

install(TARGETS              coefficients
        EXPORT               serac-targets
        DESTINATION          lib
        )
