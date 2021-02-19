# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        serac_driver
                    SOURCES     serac.cpp
                    DEPENDS_ON  serac_physics serac_coefficients
                    OUTPUT_NAME serac
                    )

blt_add_executable( NAME        serac_thermal_inverse
                    SOURCES     thermal_inverse.cpp
                    DEPENDS_ON  serac_physics serac_coefficients
                    OUTPUT_NAME thermal_inverse
                    )

if (ENABLE_TESTS)
    # Make sure running serac driver doesn't completely fail
    blt_add_test(NAME          serac_driver_default
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -i ${CMAKE_CURRENT_SOURCE_DIR}/../../data/input_files/default.lua
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_help
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac --help 
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_docs
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -d -i ${CMAKE_CURRENT_SOURCE_DIR}/../../data/input_files/default.lua
                 NUM_MPI_TASKS 1 )
endif()

install( TARGETS serac_driver
         RUNTIME DESTINATION bin
         )
