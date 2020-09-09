# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME       serac
                    SOURCES    serac.cpp
                    DEPENDS_ON common infrastructure solvers coefficients mfem mpi cli11
                    )

install( TARGETS serac
         RUNTIME DESTINATION bin
         )
