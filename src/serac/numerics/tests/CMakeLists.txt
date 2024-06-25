# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(test_dependencies gtest serac_numerics serac_boundary_conditions)

set(numerics_serial_tests
    equationsolver.cpp
    operator.cpp
    odes.cpp
    )

serac_add_tests( SOURCES ${numerics_serial_tests}
                 DEPENDS_ON ${test_dependencies}
                 NUM_MPI_TASKS 1)

if(PETSC_FOUND)
    serac_add_tests(SOURCES       equationsolver_petsc.cpp
                    DEPENDS_ON    ${test_dependencies}
                    NUM_MPI_TASKS 1)
endif()
