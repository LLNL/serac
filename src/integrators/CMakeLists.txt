# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(integrators_sources
    hyperelastic_traction_integrator.cpp
    inc_hyperelastic_integrator.cpp    
    )

set(integrators_headers
    hyperelastic_traction_integrator.hpp
    inc_hyperelastic_integrator.hpp    
    )

blt_add_library(
    NAME        integrators
    SOURCES     ${integrators_sources}
    HEADERS     ${integrators_headers}
    DEPENDS_ON  mfem mpi
    #DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis 
    )

target_include_directories(integrators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
