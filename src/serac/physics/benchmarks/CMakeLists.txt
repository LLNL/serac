# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(benchmark_depends serac_physics)

set(benchmark_targets
    benchmark_functional
    benchmark_solid_nonlinear_solve
    benchmark_thermal
    )

# Create executable for each benchmark
foreach(benchmark ${benchmark_targets})
    blt_add_executable(NAME ${benchmark}
                       SOURCES ${benchmark}.cpp
                       DEPENDS_ON ${benchmark_depends}
                       OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                       FOLDER serac/benchmarks
                       )
endforeach()

