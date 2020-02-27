# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME serac
                    SOURCES serac.cpp
                    DEPENDS_ON solvers coefficients mfem mpi
                    )

install( TARGETS serac
         RUNTIME DESTINATION bin
         )
