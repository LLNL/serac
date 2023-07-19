# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(benchmark_dependencies serac_physics)

blt_add_executable(NAME benchmark_thermal
                   SOURCES benchmark_thermal.cpp
                   DEPENDS_ON ${benchmark_dependencies}
                   OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                   FOLDER serac/benchmarks)
