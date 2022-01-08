# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(benchmark_dependencies serac_physics)
blt_list_append(TO benchmark_dependencies ELEMENTS adiak IF ${SERAC_USE_ADIAK})
blt_list_append(TO benchmark_dependencies ELEMENTS caliper IF ${SERAC_USE_CALIPER})
blt_list_append(TO benchmark_dependencies ELEMENTS mpi IF ${ENABLE_MPI})

blt_add_executable(NAME benchmark_serac_thermal_functional
                   SOURCES benchmark_serac_thermal_functional.cpp
                   DEPENDS_ON ${benchmark_dependencies}
                   OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                   FOLDER serac/benchmarks)

target_include_directories(benchmark_serac_thermal_functional
                           SYSTEM PRIVATE ${adiak_INCLUDE_DIRS})
