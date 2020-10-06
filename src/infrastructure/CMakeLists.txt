# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(infrastructure_headers
    cli.hpp
    initialize.hpp
    input.hpp
    logger.hpp
    profiling.hpp
    terminator.hpp
    )

set(infrastructure_sources
    cli.cpp
    initialize.cpp
    input.cpp
    logger.cpp
    profiling.cpp
    terminator.cpp
    )

set(infrastructure_depends axom fmt mpi cli11)
blt_list_append( TO infrastructure_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

blt_add_library(
    NAME        infrastructure
    HEADERS     ${infrastructure_headers}
    SOURCES     ${infrastructure_sources}
    DEPENDS_ON  ${infrastructure_depends}
    )

target_include_directories(infrastructure PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${infrastructure_headers} DESTINATION include/infrastructure )

install(TARGETS              infrastructure
        EXPORT               serac-targets
        DESTINATION          lib
        )
