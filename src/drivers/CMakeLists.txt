blt_add_executable( NAME serac
                    SOURCES serac.cpp
                    DEPENDS_ON mpi mfem hypre blas lapack 
                    solvers coefficients
                    )

install( TARGETS serac
         RUNTIME DESTINATION bin
         )
