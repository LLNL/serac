# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(physics_operators_headers odes.hpp stdfunction_operator.hpp)

set(physics_operators_depends infrastructure physics_utilities numerics fmt mfem mpi axom)

add_library(physics_operators ${physics_operators_headers})

set_target_properties(physics_operators PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(physics_operators INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
        )

install(FILES ${physics_operators_headers} DESTINATION include/physics/operators )

install(TARGETS              physics_operators
        EXPORT               serac-targets
        DESTINATION          lib
        )
