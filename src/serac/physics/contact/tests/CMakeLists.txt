if (TRIBOL_FOUND)

      set(test_dependencies serac_physics serac gtest tribol)

      set(tribol_tests
            serac_tribol_smoke.cpp
            serac_tribol_patch.cpp
            serac_tribol_beam.cpp)

      serac_add_tests( SOURCES ${tribol_tests}
                       DEPENDS_ON ${test_dependencies}
                       NUM_MPI_TASKS 1)
                     
endif()
