# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(test_dependencies gtest serac_numerics serac_mesh serac_boundary_conditions)

set(mesh_parallel_tests
    mesh_test.cpp
    )

serac_add_tests( SOURCES ${mesh_parallel_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 2)

set(mesh_serial_tests
    mesh_generation.cpp
    )

serac_add_tests( SOURCES ${mesh_serial_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 1)
