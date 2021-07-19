# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

if (ENABLE_GTEST)

    set(test_dependencies serac_physics serac_coefficients test_utils)
    blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

    set(numerics_parallel_tests
        serac_mesh.cpp
        )

    foreach(filename ${numerics_parallel_tests})
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

    set(numerics_serial_tests
        expr_templates.cpp
        mesh_generation.cpp
        )

    foreach(filename ${numerics_serial_tests})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${test_dependencies}
                           FOLDER      serac/tests )
        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS 1 )
    endforeach()

    if(ENABLE_BENCHMARKS)
        blt_add_executable( NAME        benchmark_expr_templates
                            SOURCES     benchmark_expr_templates.cpp
                            DEPENDS_ON  gbenchmark ${test_dependencies}
                            FOLDER      serac/tests)
        blt_add_benchmark(  NAME        benchmark_expr_templates
                            COMMAND     benchmark_expr_templates "--benchmark_min_time=0.0 --v=3 --benchmark_format=console"
                            NUM_MPI_TASKS 4)
    endif()

endif()