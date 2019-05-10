blt_add_executable( NAME flobat
                    SOURCES flobat.cpp
                    DEPENDS_ON mpi mfem hypre blas lapack parmetis
                    metis solvers coefficients
                    )

install( TARGETS flobat
         RUNTIME DESTINATION bin
         )
