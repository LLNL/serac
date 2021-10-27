# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(infrastructure_headers
    accelerator.hpp
    cli.hpp
    initialize.hpp
    input.hpp
    logger.hpp
    output.hpp
    profiling.hpp
    terminator.hpp
    variant.hpp
    )

set(infrastructure_sources
    accelerator.cpp
    cli.cpp
    initialize.cpp
    input.cpp
    logger.cpp
    output.cpp
    profiling.cpp
    terminator.cpp
    )

set(infrastructure_depends axom fmt ascent::ascent_mpi mpi cli11 mfem)
blt_list_append( TO infrastructure_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

blt_add_library(
    NAME        serac_infrastructure
    HEADERS     ${infrastructure_headers}
    SOURCES     ${infrastructure_sources}
    DEPENDS_ON  ${infrastructure_depends}
    )

if (NOT AXOM_DIR)
    # This really shouldn't be needed but it keeps getting dropped from axom.
    # It certainly shouln't go here either.
    target_include_directories(serac_infrastructure SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../axom/src>
        )
endif()

target_include_directories(serac_infrastructure PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${infrastructure_headers} DESTINATION include/serac/infrastructure )

install(TARGETS              serac_infrastructure
        EXPORT               serac-targets
        DESTINATION          lib
        )
