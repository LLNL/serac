# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

if (ENABLE_GTEST)

    set(test_dependencies serac_physics serac_coefficients test_utils)
    blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

    blt_add_library(
        NAME        test_utils
        SOURCES     test_utilities.cpp
        HEADERS     test_utilities.hpp
        DEPENDS_ON  serac_physics gtest
        )

    set(solver_tests
        serac_solid.cpp
        serac_solid_reuse_mesh.cpp
        serac_thermal_solver.cpp
        serac_thermal_structural_solver.cpp
        )

    foreach(filename ${solver_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${test_dependencies}
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 2 )
    endforeach()

    set(solver_utility_tests
        serac_dtor.cpp
        serac_boundary_cond.cpp
        serac_newmark_test.cpp)

    foreach(filename ${solver_utility_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${test_dependencies}
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 2 )
    endforeach()

endif()
