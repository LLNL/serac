# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(integrators_sources
    hyperelastic_traction_integrator.cpp
    inc_hyperelastic_integrator.cpp
    wrapper_integrator.cpp
    )

set(integrators_headers
    hyperelastic_traction_integrator.hpp
    inc_hyperelastic_integrator.hpp
    wrapper_integrator.hpp
    )

set(integrators_depends mfem mpi)
blt_list_append( TO integrators_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

blt_add_library(
    NAME        integrators
    SOURCES     ${integrators_sources}
    HEADERS     ${integrators_headers}
    DEPENDS_ON  ${integrators_depends}
    #DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis 
    )

target_include_directories(integrators PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${integrators_headers} DESTINATION include/integrators )

install(TARGETS              integrators 
        EXPORT               serac-targets
        DESTINATION          lib
        )
