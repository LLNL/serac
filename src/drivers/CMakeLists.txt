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
   blt_add_test(NAME          serac_driver_default
                COMMAND       ${PROJECT_BINARY_DIR}/bin/serac -o default -i ${input_files_dir}/default.lua
                NUM_MPI_TASKS 1 )

   blt_add_test(NAME          serac_driver_default_parallel
                COMMAND       ${PROJECT_BINARY_DIR}/bin/serac -o default_parallel -i ${input_files_dir}/default.lua
                NUM_MPI_TASKS 2 )

    blt_add_test(NAME          serac_driver_help
                 COMMAND       ${PROJECT_BINARY_DIR}/bin/serac --help 
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_docs
                 COMMAND       ${PROJECT_BINARY_DIR}/bin/serac -o default_docs -d -i ${input_files_dir}/default.lua
                 NUM_MPI_TASKS 1 )
endif()

install( TARGETS serac_driver
         RUNTIME DESTINATION bin
         )
