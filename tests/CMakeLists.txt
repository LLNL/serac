#
# Tests for SERAC 
#

set( serac_tests
     serac_hyperelasticity_solver.cpp
     )

foreach ( test ${serac_tests} )

    get_filename_component( test_name ${test} NAME_WE )

    blt_add_executable(NAME ${test_name}_test
                       SOURCES ${test}
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON solvers mfem mpi parmetis hypre gtest
                       )                      

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}_test
                  NUM_MPI_TASKS 4
                  )

endforeach()

