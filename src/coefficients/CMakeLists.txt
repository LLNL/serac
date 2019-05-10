set(coefficients_sources
    flobat_scalar_coefficient.cpp
    )

set(coefficients_headers
    flobat_scalar_coefficient.hpp
    )

blt_add_library(
    NAME        coefficients
    SOURCES     ${coefficients_sources}
    HEADERS     ${coefficients_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis models
    )

target_include_directories(coefficients PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
