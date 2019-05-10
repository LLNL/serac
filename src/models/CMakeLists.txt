set(models_sources
    parameters.cpp
    )

set(models_headers
    parameters.hpp
    )

blt_add_library(
    NAME        models
    SOURCES     ${models_sources}
    HEADERS     ${models_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis
    )

target_include_directories(models PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
