# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(physics_operators_headers odes.hpp stdfunction_operator.hpp)
set(physics_operators_sources odes.cpp)
set(physics_operators_depends infrastructure physics_utilities numerics fmt mfem mpi axom)

blt_add_library(
   NAME        physics_operators
   HEADERS     ${physics_operators_headers}
   SOURCES     ${physics_operators_sources}
   DEPENDS_ON  ${physics_operators_depends}
   )

install(FILES ${physics_operators_headers} DESTINATION include/serac/physics/operators )

install(TARGETS              physics_operators
        EXPORT               serac-targets
        DESTINATION          lib
        )
