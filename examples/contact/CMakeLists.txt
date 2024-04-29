# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND)
    blt_add_executable( NAME        contact_beam_bending
                        SOURCES     beam_bending.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics serac_mesh
                        )
    blt_add_executable( NAME        contact_ironing
                        SOURCES     ironing.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics serac_mesh
                        )
    blt_add_executable( NAME        contact_sphere
                        SOURCES     sphere.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics serac_mesh
                        )
    blt_add_executable( NAME        contact_twist
                        SOURCES     twist.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics serac_mesh
                        )
endif()
