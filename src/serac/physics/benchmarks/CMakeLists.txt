# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(benchmark_depends serac_physics)

blt_add_executable(NAME benchmark_thermal
                   SOURCES benchmark_thermal.cpp
                   DEPENDS_ON ${benchmark_depends}
                   OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                   FOLDER serac/benchmarks)

blt_add_executable(NAME benchmark_functional
                   SOURCES benchmark_functional.cpp
                   DEPENDS_ON ${benchmark_depends}
                   OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                   FOLDER serac/benchmarks)
