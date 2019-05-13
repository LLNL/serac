set(boundary_sources
    BCManager.cpp
    BCData.cpp
    )

set(boundary_headers
    BCManager.hpp
    BCData.hpp
    )

blt_add_library(
    NAME        boundary
    SOURCES     ${boundary_sources}
    HEADERS     ${boundary_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis 
    )

target_include_directories(boundary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
