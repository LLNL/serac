# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(linalg_sources
    backtracking_newton.cpp
    )

set(linalg_headers
    backtracking_newton.hpp 
   )

blt_add_library(
    NAME        linalg 
    SOURCES     ${linalg_sources}
    HEADERS     ${linalg_headers}
    DEPENDS_ON  mpi mfem hypre blas lapack parmetis metis  
    )

target_include_directories(linalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
