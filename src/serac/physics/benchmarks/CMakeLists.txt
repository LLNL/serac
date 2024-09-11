# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(physics_benchmark_depends serac_physics)

set(physics_benchmark_targets
    physics_benchmark_functional
    physics_benchmark_solid_nonlinear_solve
    physics_benchmark_thermal
    )

# Create executable for each benchmark
foreach(physics_benchmark ${physics_benchmark_targets})
    blt_add_executable(NAME ${physics_benchmark}
                       SOURCES ${physics_benchmark}.cpp
                       DEPENDS_ON ${physics_benchmark_depends}
                       OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                       FOLDER serac/benchmarks
                       )

    # Add benchmarks with various task counts
    foreach(task_count 1 4 16)
        blt_add_benchmark(NAME ${physics_benchmark}_${task_count}_task_count
                          COMMAND ${physics_benchmark}
                          NUM_MPI_TASKS ${task_count}
                          )
    endforeach()
endforeach()
