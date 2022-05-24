# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        liquid_crystal_elastomer_mechanical_with_input_file
                    SOURCES     with_input_file.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_coefficients
                    )

blt_add_executable( NAME        liquid_crystal_elastomer_mechanical_without_input_file
                    SOURCES     without_input_file.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_coefficients
                    )

blt_add_test(NAME          liquid_crystal_elastomer_mechanical_with_input_file
             COMMAND       liquid_crystal_elastomer_mechanical_with_input_file -o liquid_crystal_elastomer_example -i ${CMAKE_CURRENT_SOURCE_DIR}/liquid_crystal_elastomer.lua
             NUM_MPI_TASKS 1 )

blt_add_test(NAME          liquid_crystal_elastomer_mechanical_without_input_file
             COMMAND       liquid_crystal_elastomer_mechanical_without_input_file
             NUM_MPI_TASKS 1 )


