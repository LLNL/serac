# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        serac_driver
                    SOURCES     serac.cpp
                    DEPENDS_ON  numerics infrastructure physics coefficients mfem mpi
                    OUTPUT_NAME serac
                    )

install( TARGETS serac_driver
         RUNTIME DESTINATION bin
         )
