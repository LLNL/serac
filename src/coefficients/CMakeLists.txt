set(coefficients_sources
    loading_functions.cpp
    )

set(coefficients_headers
    loading_functions.hpp
    )

blt_add_library(
    NAME        coefficients
    SOURCES     ${coefficients_sources}
    HEADERS     ${coefficients_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis
    )

target_include_directories(coefficients PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
