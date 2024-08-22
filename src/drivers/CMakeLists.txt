# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        serac_driver
                    SOURCES     serac.cpp
                    DEPENDS_ON  serac_physics serac_mesh
                    OUTPUT_NAME serac
                    )

if (SERAC_ENABLE_TESTS)
    set(input_files_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../data/input_files)

    # Make sure running serac driver doesn't completely fail
#TODO: Turn these back on when input files work with the new physics modules
#    blt_add_test(NAME          serac_driver_default
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o default -i ${input_files_dir}/default.lua
#                 NUM_MPI_TASKS 1 )

#    blt_add_test(NAME          serac_driver_default_parallel
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o default_parallel -i ${input_files_dir}/default.lua
#                 NUM_MPI_TASKS 2 )

#    blt_add_test(NAME          serac_driver_no_thermal
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o default_no_thermal -i ${input_files_dir}/tests/solid/qs_linear.lua
#                 NUM_MPI_TASKS 1 )

#    blt_add_test(NAME          serac_driver_no_solid
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o default_no_solid -i ${input_files_dir}/tests/heat_transfer/static_solve.lua
#                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_help
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac --help 
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_docs
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o default_docs -d -i ${input_files_dir}/default.lua
                 NUM_MPI_TASKS 1 )
endif()

install( TARGETS serac_driver
         RUNTIME DESTINATION bin
         )
