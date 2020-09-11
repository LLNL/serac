# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(math)
add_subdirectory(solvers)
add_subdirectory(integrators)
add_subdirectory(coefficients)
add_subdirectory(drivers)
add_subdirectory(infrastructure)

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()
