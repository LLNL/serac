set(integrators_sources
    hyperelastic_traction_integrator.cpp
    )

set(integrators_headers
    hyperelastic_traction_integrator.hpp
    )

blt_add_library(
    NAME        integrators
    SOURCES     ${integrators_sources}
    HEADERS     ${integrators_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis 
    )

target_include_directories(integrators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
