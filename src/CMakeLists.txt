# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(serac)
add_subdirectory(drivers)

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
