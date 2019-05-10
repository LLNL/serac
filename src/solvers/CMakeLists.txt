set(solvers_sources
    dg_diffusion_solver.cpp
    dg_convection_solver.cpp
    cg_diffusion_solver.cpp
    cg_convection_solver.cpp
    coupled_potential_solver.cpp
    )

set(solvers_headers
    dg_diffusion_solver.hpp
    dg_convection_solver.hpp
    cg_diffusion_solver.hpp
    cg_convection_solver.hpp    
    coupled_potential_solver.hpp
    )

blt_add_library(
    NAME        solvers
    SOURCES     ${solvers_sources}
    HEADERS     ${solvers_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis integrators
    )

target_include_directories(solvers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
