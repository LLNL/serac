# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies gtest serac_numerics serac_boundary_conditions serac_state)
blt_list_append( TO test_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER} )

set(numerics_serial_tests
    expr_templates.cpp
    serac_operator.cpp
    serac_odes.cpp
    )

serac_add_tests( SOURCES ${numerics_serial_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 1)

if(ENABLE_BENCHMARKS)
    blt_add_executable( NAME        benchmark_expr_templates
                        SOURCES     benchmark_expr_templates.cpp
                        DEPENDS_ON  gbenchmark ${test_dependencies}
                        FOLDER      serac/tests)
    blt_add_benchmark(  NAME        benchmark_expr_templates
                        COMMAND     benchmark_expr_templates "--benchmark_min_time=0.0 --v=3 --benchmark_format=console"
                        NUM_MPI_TASKS 4)
endif()
