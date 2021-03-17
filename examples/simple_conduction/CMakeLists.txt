# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        simple_conduction_with_input_file
                    SOURCES     with_input_file.cpp
                    DEPENDS_ON  serac_physics serac_coefficients
                    )

blt_add_executable( NAME        simple_conduction_without_input_file
                    SOURCES     without_input_file.cpp
                    DEPENDS_ON  serac_physics serac_coefficients
                    )
