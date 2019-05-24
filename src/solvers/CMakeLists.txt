set(solvers_sources
    hyperelastic_solver.cpp
    )

set(solvers_headers
    hyperelastic_solver.hpp
    )

blt_add_library(
    NAME        solvers
    SOURCES     ${solvers_sources}
    HEADERS     ${solvers_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis integrators
    )

target_include_directories(solvers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
